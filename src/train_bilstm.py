import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer
from datasets import load_dataset

from utils import set_seed 
from data_pipeline import EmotionDataset, DataPipelineManager

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

    @property
    def vocab_size(self):
        return len(self.vocab)

def build_tokenizer(model_type, train_texts=None):
    if model_type == "DistilBERT":
        return AutoTokenizer.from_pretrained('distilbert-base-uncased')
    elif model_type == "BiLSTM":
        custom_tokenizer = CustomSimpleTokenizer(min_freq=2)
        custom_tokenizer.build_vocab(train_texts)
        return custom_tokenizer
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
        # text shape: [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        
        # Pack sequence (Optimization for RNNs)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        return self.fc(hidden)

# ---------------------------------------------------------
# 2. TRAINING LOOP 
# ---------------------------------------------------------
def train_model(train_loader, val_loader, tokenizer, device, model_type="BiLSTM", use_class_weights=False):
    
    wandb.init(
        entity="nlp-emotion",
        project="nlp-emotion-classification", 
        name=f"{model_type}_(Baseline-class-weights)_Run", 
        config={
            "architecture": model_type,
            "learning_rate": 0.001 if model_type == "BiLSTM" else 2e-5, 
            "epochs": 10, # Run 10 epochs for testing
            "batch_size": 64,
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": 100,
            "hidden_dim": 256,
            "n_layers": 2,
            "output_dim": 28
        }
    )
    config = wandb.config

    print(f"Building {model_type} architecture...")
    if model_type == "BiLSTM":
        pad_idx = tokenizer.pad_token_id
        model = BiLSTMClassifier(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            n_layers=config.n_layers,
            bidirectional=True,
            dropout=0.5,
            pad_idx=pad_idx
        ).to(device)
    elif model_type == "DistilBERT":
        pad_idx = tokenizer.pad_token_id
        from task5_lstm_distilbert_experiment import DistilBert_Advanced
        model = DistilBert_Advanced(num_classes=config.output_dim).to(device)
    else:
        raise ValueError("Invalid model name!")
    
    if use_class_weights:
        pos_weights = torch.load("data/processed/pos_weights.pt").to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_val_f1 = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    print("Starting training loop...")
    for epoch in range(config.epochs):

        # ================= TRAIN PHASE =================
        model.train()
        train_loss = 0
        
        for texts, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            
            if model_type == "BiLSTM":
                # Dùng bộ tokenizer tự chế để chuyển text sang tensor số
                tokenized = [torch.tensor(tokenizer.text_to_ids(t)) for t in texts]
                # Xử lý các câu rỗng nếu có rác phát sinh
                tokenized = [t if len(t) > 0 else torch.tensor([tokenizer.unk_token_id]) for t in tokenized]
                
                input_ids = nn.utils.rnn.pad_sequence(tokenized, batch_first=True, padding_value=pad_idx).to(device)
                text_lengths = torch.tensor([len(t) for t in tokenized])
                logits = model(input_ids, text_lengths)

            elif model_type == "DistilBERT":
                inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                logits = model(input_ids, attention_mask=attention_mask)
                
            loss = criterion(logits, labels.float()) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(texts) 
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # ================= VALIDATION PHASE =================
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in val_loader:
                labels = labels.to(device)
                
                if model_type == "BiLSTM":
                    tokenized = [torch.tensor(tokenizer.text_to_ids(t)) for t in texts]
                    tokenized = [t if len(t) > 0 else torch.tensor([tokenizer.unk_token_id]) for t in tokenized]
                    input_ids = nn.utils.rnn.pad_sequence(tokenized, batch_first=True, padding_value=pad_idx).to(device)
                    text_lengths = torch.tensor([len(t) for t in tokenized])
                    logits = model(input_ids, text_lengths)
                    
                elif model_type == "DistilBERT":
                    inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    logits = model(input_ids, attention_mask=attention_mask)
                
                loss = criterion(logits, labels.float())
                val_loss += loss.item() * len(texts)
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader.dataset)

        val_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            checkpoint_path = f"checkpoints/{model_type}_best.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_macro_f1,
                "val_micro_f1": val_micro_f1,
            }, checkpoint_path)
        
        
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
        
        print(f"--- Epoch {epoch+1} ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Macro F1: {val_macro_f1:.4f} | Val Micro F1: {val_micro_f1:.4f}\n")
    
    print(f"Saved best model at epoch {epoch+1} (Macro F1: {best_val_f1:.4f})")
    wandb.finish()
    print("Training completed! Check your W&B dashboard for detailed metrics and visualizations.")
    return model # Return Model in order to test it on the test set after training

if __name__ == "__main__":
    # Set up reproducibility and device
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model_type = "BiLSTM"

    TRAIN_CSV = "data/processed/train.csv"
    VAL_CSV = "data/processed/val.csv"

    print("Setting up data pipeline...")
    pipeline = DataPipelineManager(
        raw_data_path=TRAIN_CSV, 
        mock_data_output_path=TRAIN_CSV
    )
    # df = pd.read_csv("data/raw/mock_data.csv")
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    print(f"-> Loaded Train set: {len(df_train)} rows | Val set: {len(df_val)} rows")

    BATCH_SIZE = 64
    train_loader = pipeline.get_pytorch_loaders(df_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = pipeline.get_pytorch_loaders(df_val, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading Tokenizer...")
    if model_type == "BiLSTM":
        tokenizer = build_tokenizer("BiLSTM", train_texts=df_train['text'].tolist())
    else:
        tokenizer = build_tokenizer("DistilBERT")

    # Train the model (BiLSTM or DistilBERT)
    train_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        tokenizer=tokenizer, 
        device=device,
        model_type=model_type,
        use_class_weights=True
    )
