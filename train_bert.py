# =====================================================================
# FILE: main.py
# ROLE: Central orchestration, experiment configuration, and structured model saving
# =====================================================================

import argparse
import os
import torch
import torch.nn as nn
import wandb
import pandas as pd
from transformers import DistilBertTokenizer, RobertaTokenizer

from utils import set_seed
from data_pipeline import DataPipelineManager
from models_bert import DistilBertEmotionClassifier, RoBERTaEmotionClassifier
from engine import train_one_epoch, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="GoEmotions Structured Framework")
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True, 
        choices=[
            "distilbert_partial_freeze", 
            "distilbert_full_tuned_weighted", 
            "distilbert_full_tuned_unweighted", 
            "roberta_full_tuned"
        ],
        help="Specify the experimental tracking setup."
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Framework Activation ===")
    print(f"Target Compute Device: {device}")

    # 1. Dataset Loading Pipeline (Structured Paths)
    train_path = os.path.join("data", "train.csv")
    val_path = os.path.join("data", "val.csv")
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    LABEL_COLUMNS = [col for col in df_train.columns if col not in ['text', 'id']]
    NUM_CLASSES = len(LABEL_COLUMNS)

    pipeline = DataPipelineManager(raw_data_path=train_path, mock_data_output_path=os.path.join("data", "mock_data.csv"))

    # 2. Strategy Mapping Configuration
    if args.experiment == "distilbert_partial_freeze":
        config = {"name": "DistilBERT_Partial_Freeze", "lr": 2e-5, "bs": 32, "model": "distilbert"}
        model = DistilBertEmotionClassifier(NUM_CLASSES, freeze_backbone=True, partial_freeze=True).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        use_weights = True
        step_size = 0.01
        
    elif args.experiment == "distilbert_full_tuned_weighted":
        config = {"name": "DistilBERT_Full_Tuned_Weighted", "lr": 2e-5, "bs": 32, "model": "distilbert"}
        model = DistilBertEmotionClassifier(NUM_CLASSES, freeze_backbone=False).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        use_weights = True
        step_size = 0.01
        
    elif args.experiment == "distilbert_full_tuned_unweighted":
        config = {"name": "DistilBERT_Full_Tuned_Unweighted", "lr": 2e-5, "bs": 32, "model": "distilbert"}
        model = DistilBertEmotionClassifier(NUM_CLASSES, freeze_backbone=False).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        use_weights = False
        step_size = 0.01
        
    elif args.experiment == "roberta_full_tuned":
        config = {"name": "RoBERTa_Full_Tuned", "lr": 1e-5, "bs": 16, "model": "roberta"}
        model = RoBERTaEmotionClassifier(NUM_CLASSES).to(device)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        use_weights = True
        step_size = 0.05

    train_loader = pipeline.get_pytorch_loaders(df_train, batch_size=config["bs"], shuffle=True)
    val_loader = pipeline.get_pytorch_loaders(df_val, batch_size=32, shuffle=False)

    # 3. Initialization of Loss and Optimizer
    wandb.init(entity="nlp-emotion", project="nlp-emotion-classification", name=config["name"], config=config)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    
    weight_path = os.path.join("weights", "pos_weights.pt")
    if use_weights and os.path.exists(weight_path):
        print(f"-> Status: Loading class weights from '{weight_path}'.")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.load(weight_path).to(device))
    else:
        print("-> Status: Initializing standard unweighted BCE Loss.")
        criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 4. Training Loop Execution
    best_macro_f1 = 0.0
    os.makedirs("weights", exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss = train_one_epoch(model, train_loader, tokenizer, optimizer, criterion, scaler, device)
        val_loss, macro_f1, micro_f1, prec, rec, top_5_df, thresholds = evaluate_model(
            model, val_loader, tokenizer, criterion, device, LABEL_COLUMNS, step=step_size
        )

        scheduler.step(macro_f1)
        print(f"Val Loss: {val_loss:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")

        metrics_log = {"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "macro_f1": macro_f1, "micro_f1": micro_f1}
        metrics_log[f"top_5_hardest_epoch_{epoch+1}"] = wandb.Table(dataframe=top_5_df)
        wandb.log(metrics_log)

        # Secure Checkpoint Saving inside weights/ folder
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "thresholds": thresholds,
                "label_names": LABEL_COLUMNS
            }
            
            checkpoint_path = os.path.join("weights", f"best_{args.experiment}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f">> Success: Better model isolated. Checkpoint saved at: {checkpoint_path}")

    wandb.finish()

if __name__ == "__main__":
    main()
