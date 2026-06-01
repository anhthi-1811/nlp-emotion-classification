import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer
#from task5_lstm_distilbert_experiment import DistilBert_Advanced

from utils import set_seed 
from data_pipeline import EmotionDataset, DataPipelineManager

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
def train_model(train_loader, val_loader, tokenizer, device, model_type="BiLSTM"):
    
    wandb.init(
        entity="nlp-emotion",
        project="nlp-emotion-classification", 
        name=f"{model_type}_Run", 
        config={
            "architecture": model_type,
            "learning_rate": 0.001 if model_type == "BiLSTM" else 2e-5, 
            "epochs": 10, # Run 10 epochs for testing
            "batch_size": 16,
            "vocab_size": tokenizer.vocab_size, # Update to exact HF vocabulary size
            "embedding_dim": 100,
            "hidden_dim": 256,
            "n_layers": 2,
            "output_dim": 6 
        }
    )
    config = wandb.config

    print(f"Building {model_type} ...")
    if model_type == "BiLSTM":
        model = BiLSTMClassifier(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            n_layers=config.n_layers,
            bidirectional=True,
            dropout=0.5,
            pad_idx=tokenizer.pad_token_id # Sync PAD index with Tokenizer
        ).to(device)
    elif model_type == "DistilBERT":
        model = DistilBert_Advanced(num_classes=config.output_dim).to(device)
    else:
        raise ValueError("Invalid model name!")
    
    print("Loading class weights for imbalance handling...")
    class_weights = torch.load("data/processed/class_weights.pt").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight= class_weights)

    print("Starting training loop...")
    for epoch in range(config.epochs):
        
        # ================= TRAIN PHASE =================
        model.train()
        train_loss = 0
        
        for texts, labels in train_loader:
            
            # Process text using Tokenizer
            inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if model_type == "DistilBERT":
                logits = model(input_ids, attention_mask=attention_mask)
            elif model_type == "BiLSTM":
                # Count actual tokens (Sum of 1s in mask)
                text_lengths = attention_mask.sum(dim=1).to('cpu') 
                logits = model(input_ids, text_lengths)
                
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # ================= VALIDATION PHASE =================
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts, labels in val_loader:
                
                # Process text using Tokenizer
                inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                labels = labels.to(device)
                
                if model_type == "DistilBERT":
                    logits = model(input_ids, attention_mask=attention_mask)
                elif model_type == "BiLSTM":
                    text_lengths = attention_mask.sum(dim=1).to('cpu')
                    logits = model(input_ids, text_lengths)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) 
        
        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
            "val_f1_score": val_f1
        })
        
        print(f"--- Epoch {epoch+1} ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f} | Val F1-Score: {val_f1:.4f}\n")
    
    wandb.finish()
    print("Training completed! Check your W&B dashboard for detailed metrics and visualizations.")

if __name__ == "__main__":
    # Set up reproducibility and device
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    print("Setting up data pipeline...")
    pipeline = DataPipelineManager(
        raw_data_path="data/processed/train.csv", 
        mock_data_output_path="data/raw/mock_data_temp.csv"
    )
    # df = pd.read_csv("data/raw/mock_data.csv")
    df = pd.read_csv("data/processed/train.csv")
        
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    BATCH_SIZE = 32
    train_loader = pipeline.get_pytorch_loaders(df_train, batch_size=BATCH_SIZE)
    val_loader = pipeline.get_pytorch_loaders(df_val, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Train the model (BiLSTM or DistilBERT)
    train_model(
        train_loader=train_loader, 
        val_loader=val_loader, 
        tokenizer=tokenizer, 
        device=device,
        model_type="BiLSTM"
    )