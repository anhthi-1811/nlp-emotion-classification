# =====================================================================
# FILE: train_bilstm.py
# ROLE: Build and train a Bi-LSTM model for multi-label emotion classification on the GoEmotions dataset.
# =====================================================================
import torch
import torch.nn as nn
import wandb
import pandas as pd
import numpy as np
import re
import os
from collections import Counter
from sklearn.metrics import f1_score
import gensim.downloader as api

from utils import set_seed
from data_pipeline import EmotionDataset, DataPipelineManager
from metrics import evaluate_goemotions

class CustomSimpleTokenizer:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.vocab = {"<unk>": 0, "<pad>": 1}
        self.pad_token_id = 1
        self.unk_token_id = 0

    def tokenize_fn(self, text):
        # Basic tokenization using regex: convert to lowercase and extract alphanumeric tokens
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocab(self, texts):
        token_counter = Counter()
        for text in texts:
            tokens = self.tokenize_fn(text)
            token_counter.update(tokens)
        
        # Keep only tokens with a frequency >= min_freq
        idx = len(self.vocab)
        for token, freq in token_counter.items():
            if freq >= self.min_freq and token not in self.vocab:
                self.vocab[token] = idx
                idx += 1
                
    def text_to_ids(self, text):
        # Convert a text string into a list of numerical token IDs based on the vocabulary
        tokens = self.tokenize_fn(text)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def get_itos(self):
        # Return a list mapping from token IDs back to tokens (inverse of vocab)
        itos = [""] * len(self.vocab)
        for token, idx in self.vocab.items():
            itos[idx] = token
        return itos
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
def build_glove_embedding_matrix(tokenizer, glove_model, embedding_dim=300):
    vocab_size       = tokenizer.vocab_size
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)
    itos             = tokenizer.get_itos()
    found            = 0
    for idx, token in enumerate(itos):
        if token in glove_model:
            embedding_matrix[idx] = torch.tensor(glove_model[token])
            found += 1
    print(f"GloVe coverage: {found}/{vocab_size} tokens ({found/vocab_size*100:.1f}%)")
    return embedding_matrix

# ---------------------------------------------------------
# 1. MODEL ARCHITECTURE (Bi-LSTM)
# ---------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded        = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed_embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)
    
    def load_pretrained_embeddings(self, embedding_matrix):
        self.embedding.weight.data.copy_(embedding_matrix)
        print(f"Loaded pretrained embeddings: {embedding_matrix.shape}")

# ---------------------------------------------------------
# 2. TRAINING LOOP 
# ---------------------------------------------------------
def train_model(train_loader, val_loader, tokenizer, device, label_names, use_class_weights=False,
                embedding_matrix=None, hidden_dim=256, dropout=0.5, weight_decay=1e-5,
                epochs=10, use_scheduler=False, use_grad_clipping=False, 
                early_stopping=False, patience=3, monitor_metric='f1'):
    # --- W&B Run Name ---
    run_name = "BiLSTM_w" if use_class_weights else "BiLSTM_no_w"
    if embedding_matrix is not None: run_name += "+GloVe"
    if hidden_dim != 256:            run_name += f"+HD{hidden_dim}"
    if epochs != 10:                 run_name += f"+Ep{epochs}"
    if use_scheduler:                run_name += "+Sched"
    if use_grad_clipping:            run_name += "+Clip"

    embedding_dim = embedding_matrix.shape[1] if embedding_matrix is not None else 100

    wandb.init(
        entity="nlp-emotion",
        project="nlp-emotion-classification", 
        name=run_name, 
        config={
            "architecture": "BiLSTM",
            "learning_rate": 0.001, 
            "epochs": epochs, 
            "batch_size": 64,
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "n_layers": 2,
            "output_dim": 28,
            "use_glove": embedding_matrix is not None,
            "use_class_weights": use_class_weights,
            "use_scheduler":     use_scheduler,
            "use_grad_clipping": use_grad_clipping,
            "early_stopping":    early_stopping,
            "patience":          patience if early_stopping else None,
            "dropout":           dropout,        
            "weight_decay":      weight_decay,   
            "monitor_metric":    monitor_metric
        }
    )
    config = wandb.config
    pad_idx = tokenizer.pad_token_id

    # --- Build Model ---
    print(f"Building Bi-LSTM architecture...")
    model = BiLSTMClassifier(
        vocab_size    = config.vocab_size,
        embedding_dim = config.embedding_dim,
        hidden_dim    = config.hidden_dim,
        output_dim    = config.output_dim,
        n_layers      = config.n_layers,
        bidirectional = True,
        dropout       = config.dropout,
        pad_idx       = pad_idx
    ).to(device)
 
    if embedding_matrix is not None:
        model.load_pretrained_embeddings(embedding_matrix)
        model.embedding.weight.requires_grad = False  # freeze GloVe
    
    # --- Loss, Optimizer, Scheduler ---
    if use_class_weights:
        pos_weights = torch.load("data/processed/pos_weights.pt").to(device)
        assert pos_weights.shape == (28,), \
            f"pos_weights shape sai: {pos_weights.shape}, expected (28,)"
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
    # --- Tracking ---
    best_val_loss = float('inf')
    best_val_f1   = 0.0  
    best_epoch     = 0
    patience_count = 0

    print("Starting training loop...")
    for epoch in range(config.epochs):

        # ================= TRAIN PHASE =================
        model.train()
        train_loss = 0
        
        for texts, labels in train_loader:
            labels       = labels.to(device)
            optimizer.zero_grad()
            tokenized    = [torch.tensor(tokenizer.text_to_ids(t)) for t in texts]
            tokenized    = [t if len(t) > 0 else torch.tensor([tokenizer.unk_token_id]) for t in tokenized]
            input_ids    = nn.utils.rnn.pad_sequence(tokenized, batch_first=True, padding_value=pad_idx).to(device)
            text_lengths = torch.tensor([len(t) for t in tokenized])
            logits       = model(input_ids, text_lengths)
            loss         = criterion(logits, labels.float())
            loss.backward()
            if use_grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(texts) 
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # ================= VALIDATION PHASE =================
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in val_loader:
                labels       = labels.to(device)
                tokenized    = [torch.tensor(tokenizer.text_to_ids(t)) for t in texts]
                tokenized    = [t if len(t) > 0 else torch.tensor([tokenizer.unk_token_id]) for t in tokenized]
                input_ids    = nn.utils.rnn.pad_sequence(tokenized, batch_first=True, padding_value=pad_idx).to(device)
                text_lengths = torch.tensor([len(t) for t in tokenized])
                logits       = model(input_ids, text_lengths)
                val_loss    += criterion(logits, labels.float()).item() * len(texts)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
 
        all_probs    = np.vstack(all_probs)
        all_labels   = np.vstack(all_labels)
        avg_val_loss = val_loss / len(val_loader.dataset)

        # --- Metrics (dùng metrics.py) ---
        val_macro_f1, val_micro_f1, val_precision, val_recall, _ = evaluate_goemotions(
            y_true       = all_labels,
            y_pred_probs = all_probs,
            label_names  = label_names,
            threshold    = 0.5,
            verbose      = False
        )

        # Scheduler step
        if use_scheduler:
            scheduler.step(val_macro_f1)
 
        # Track best epoch
        if monitor_metric == 'f1':
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_epoch  = epoch + 1
                patience_count = 0
                
                # Save checkpoint
                os.makedirs("weights", exist_ok=True)
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "label_names":      label_names,
                    "vocab":            tokenizer.vocab,        
                    "config": {
                        "vocab_size":    config.vocab_size,
                        "embedding_dim": config.embedding_dim,
                        "hidden_dim":    config.hidden_dim,
                        "n_layers":      config.n_layers,
                        "dropout":       config.dropout,
                    }
                }
                torch.save(checkpoint, f"weights/best_bilstm_{run_name}.pt")
                
            else:
                patience_count += 1
                if early_stopping:
                    print(f" No improvement (Val F1): {patience_count}/{patience}")

        elif monitor_metric == 'loss':
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_epoch  = epoch + 1
                
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_count = 0
            else:
                patience_count += 1
                if early_stopping:
                    print(f" No improvement (Val Loss): {patience_count}/{patience}")

        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_macro_f1": val_macro_f1,
            "val_micro_f1": val_micro_f1,
            "val_precision": val_precision,
            "val_recall": val_recall
        })
        
        print(f"--- Epoch {epoch+1}/{config.epochs} ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Macro F1: {val_macro_f1:.4f} | Val Micro F1: {val_micro_f1:.4f}\n")

        # Early stopping check
        if early_stopping and patience_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nBest Val Macro F1: {best_val_f1:.4f} at Epoch {best_epoch}/{config.epochs}")
    print(f"Saved checkpoint: weights/best_bilstm_{run_name}.pt")
    wandb.finish()
    print("Training completed! Check your W&B dashboard for detailed metrics and visualizations.")
    return model # Return Model in order to test it on the test set after training

# ---------------------------------------------------------
# THRESHOLD TUNING
# ---------------------------------------------------------
def tune_threshold(model, val_loader, tokenizer, device, label_names):
    pad_idx = tokenizer.pad_token_id
    model.eval()
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in val_loader:
            labels       = labels.to(device)
            tokenized    = [torch.tensor(tokenizer.text_to_ids(t)) for t in texts]
            tokenized    = [t if len(t) > 0 else torch.tensor([tokenizer.unk_token_id]) for t in tokenized]
            input_ids    = nn.utils.rnn.pad_sequence(tokenized, batch_first=True, padding_value=pad_idx).to(device)
            text_lengths = torch.tensor([len(t) for t in tokenized])
            logits       = model(input_ids, text_lengths)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
 
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
 
    print("\n--- Threshold Tuning on Val Set ---")
    best_thresh = 0.5
    best_f1     = 0.0
    for thresh in np.arange(0.2, 0.6, 0.05):
        preds  = (all_probs > thresh).astype(int)
        f1     = f1_score(all_labels, preds, average='macro', zero_division=0)
        print(f"  Threshold {thresh:.2f}: Macro F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh
 
    print(f"\nBest threshold: {best_thresh:.2f} → Val Macro F1: {best_f1:.4f}")
    return best_thresh

# ---------------------------------------------------------
# TEST SET EVALUATION
# ---------------------------------------------------------
def evaluate_on_test(model, test_loader, tokenizer, device, label_names, threshold=0.5):
    pad_idx = tokenizer.pad_token_id
    model.eval()
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            labels       = labels.to(device)
            tokenized    = [torch.tensor(tokenizer.text_to_ids(t)) for t in texts]
            tokenized    = [t if len(t) > 0 else torch.tensor([tokenizer.unk_token_id]) for t in tokenized]
            input_ids    = nn.utils.rnn.pad_sequence(tokenized, batch_first=True, padding_value=pad_idx).to(device)
            text_lengths = torch.tensor([len(t) for t in tokenized])
            logits       = model(input_ids, text_lengths)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
 
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
 
    print(f"\n{'='*50}")
    print(f"  TEST SET RESULTS (threshold={threshold:.2f})")
    print(f"{'='*50}")
    macro_f1, micro_f1, precision, recall, _ = evaluate_goemotions(
        y_true       = all_labels,
        y_pred_probs = all_probs,
        label_names  = label_names,
        threshold    = threshold
    )
    return {"macro_f1": macro_f1, "micro_f1": micro_f1,
            "precision": precision, "recall": recall}
 

if __name__ == "__main__":
    # Set up reproducibility and device
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    TRAIN_CSV = "data/processed/train.csv"
    VAL_CSV   = "data/processed/val.csv"
    TEST_CSV  = "data/processed/test.csv"
 
    print("Setting up data pipeline...")
    pipeline = DataPipelineManager(raw_data_path=TRAIN_CSV, mock_data_output_path=TRAIN_CSV)
 
    df_train = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df_val   = pd.read_csv(VAL_CSV,   encoding='utf-8-sig')
    df_test  = pd.read_csv(TEST_CSV,  encoding='utf-8-sig')
    print(f"-> Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)} rows")

    LABEL_NAMES = [col for col in df_train.columns if col not in ['text', 'id']]

    BATCH_SIZE = 64
    train_loader = pipeline.get_pytorch_loaders(df_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = pipeline.get_pytorch_loaders(df_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = pipeline.get_pytorch_loaders(df_test, batch_size=BATCH_SIZE, shuffle=False)

    # --- Tokenizer + GloVe ---
    print("Loading Tokenizer...")
    tokenizer = CustomSimpleTokenizer(min_freq=2)
    tokenizer.build_vocab(df_train['text'].tolist())
    print(f"Vocab size: {tokenizer.vocab_size}")
 
    print("Loading GloVe 300d via gensim...")
    glove_model      = api.load("glove-wiki-gigaword-300")
    embedding_matrix = build_glove_embedding_matrix(tokenizer, glove_model, embedding_dim=300)


    # -----------------------------------------------------------------------
    # EXPERIMENT 1: BASELINE (Tự học embedding 100d, Não nhỏ HD 256)
    # -----------------------------------------------------------------------
    # model = train_model(
    #     train_loader=train_loader, val_loader=val_loader,
    #     tokenizer=tokenizer, device=device, label_names=LABEL_NAMES,
    #     use_class_weights=True, embedding_matrix=None,
    #     hidden_dim=256, epochs=10, dropout=0.5, weight_decay=1e-5,
    #     use_scheduler=False, use_grad_clipping=False, early_stopping=False
    # )
 
    # -----------------------------------------------------------------------
    # EXPERIMENT 2: + GLOVE 300D (Bơm tri thức, Não nhỏ HD 256)
    # -----------------------------------------------------------------------
    # model = train_model(
    #     train_loader=train_loader, val_loader=val_loader,
    #     tokenizer=tokenizer, device=device, label_names=LABEL_NAMES,
    #     use_class_weights=True, embedding_matrix=embedding_matrix,
    #     hidden_dim=256, epochs=10, dropout=0.5, weight_decay=1e-5,
    #     use_scheduler=False, use_grad_clipping=False, early_stopping=False
    # )
 
    # -----------------------------------------------------------------------
    # EXPERIMENT 3: GloVe, Não to HD 512, Dừng theo F1
    # -----------------------------------------------------------------------
    model = train_model(
        train_loader=train_loader, val_loader=val_loader,
        tokenizer=tokenizer, device=device, label_names=LABEL_NAMES,
        use_class_weights=True, embedding_matrix=embedding_matrix,
        hidden_dim=512, epochs=30, dropout=0.5, weight_decay=1e-5,
        use_scheduler=True, use_grad_clipping=True,
        early_stopping=True, patience=5, monitor_metric='f1'
    )
 
    # -----------------------------------------------------------------------
    # EXPERIMENT 4: Tắt class_weights để so sánh với Exp 3
    # -----------------------------------------------------------------------
    # model = train_model(
    #     train_loader=train_loader, val_loader=val_loader,
    #     tokenizer=tokenizer, device=device, label_names=LABEL_NAMES,
    #     use_class_weights=False,   # <--- FALSE ĐỂ TEST SỰ THIẾU HỤT
    #     embedding_matrix=embedding_matrix,
    #     hidden_dim=512, epochs=30, dropout=0.5, weight_decay=1e-5,
    #     use_scheduler=True, use_grad_clipping=True,
    #     early_stopping=True, patience=5, monitor_metric='f1'
    # )

    # -----------------------------------------------------------------------
    # EXPERIMENT 5: GENERALIZATION CHECK (Mô hình ngoan, Dừng theo Loss)
    # -----------------------------------------------------------------------
    # model = train_model(
    #     train_loader=train_loader, val_loader=val_loader,
    #     tokenizer=tokenizer, device=device, label_names=LABEL_NAMES,
    #     use_class_weights=True, embedding_matrix=embedding_matrix,
    #     hidden_dim=256, epochs=30, dropout=0.6, weight_decay=1e-3,
    #     use_scheduler=True, use_grad_clipping=True,
    #     early_stopping=True, patience=5, monitor_metric='loss'
    # )

    # -----------------------------------------------------------------------
    # THRESHOLD TUNING + TEST EVALUATION
    # -----------------------------------------------------------------------
    best_thresh = tune_threshold(model, val_loader, tokenizer, device, LABEL_NAMES)
    evaluate_on_test(model, test_loader, tokenizer, device, LABEL_NAMES, threshold=best_thresh)
 
