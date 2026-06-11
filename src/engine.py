%%writefile engine.py
# =====================================================================
# FILE: engine.py
# ROLE: Core training execution, validation loops, and threshold tuning
# =====================================================================

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from metrics import evaluate_goemotions

def optimize_thresholds(y_true, y_prob, num_classes, step=0.01):
    """
    Performs grid search to find the optimal F1 threshold per individual class.
    """
    best_thresholds = np.full(num_classes, 0.5)
    possible_thresholds = np.arange(0.05, 0.96, step)
    
    for c in range(num_classes):
        best_f1_class = -1
        best_t = 0.5
        for t in possible_thresholds:
            preds_class = (y_prob[:, c] >= t).astype(int)
            score = f1_score(y_true[:, c], preds_class, zero_division=0)
            if score > best_f1_class:
                best_f1_class = score
                best_t = t
        best_thresholds[c] = best_t
    return best_thresholds

def adapt_probs(y_prob, thresholds, num_classes):
    """
    Normalizes multi-threshold output probabilities relative to a standardized 0.5 anchor.
    """
    y_prob_adapted = np.zeros_like(y_prob)
    for c in range(num_classes):
        mask_above = y_prob[:, c] >= thresholds[c]
        y_prob_adapted[mask_above, c] = 0.5 + (y_prob[mask_above, c] - thresholds[c]) * (0.5 / (1.0 - thresholds[c] + 1e-6))
        y_prob_adapted[~mask_above, c] = (y_prob[~mask_above, c]) * (0.5 / (thresholds[c] + 1e-6))
    return y_prob_adapted

def train_one_epoch(model, dataloader, tokenizer, optimizer, criterion, scaler, device):
    """
    Executes a single training epoch utilizing Mixed Precision (AMP).
    """
    model.train()
    total_loss = 0.0
    for texts, labels in tqdm(dataloader, desc="Training Iteration", leave=False):
        optimizer.zero_grad()
        encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, tokenizer, criterion, device, label_columns, step=0.01):
    """
    Validates model performance and computes error analysis metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds_probs, all_true_labels = [], []
    
    with torch.no_grad():
        for texts, labels in dataloader:
            encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(outputs).float().cpu().numpy()
            all_preds_probs.append(probs)
            all_true_labels.append(labels.float().cpu().numpy())

    y_true = np.vstack(all_true_labels)
    y_prob = np.vstack(all_preds_probs)
    num_classes = len(label_columns)
    
    thresholds = optimize_thresholds(y_true, y_prob, num_classes, step=step)
    y_prob_adapted = adapt_probs(y_prob, thresholds, num_classes)
    
    macro_f1, micro_f1, prec, rec, top_5_df = evaluate_goemotions(
        y_true=y_true, y_pred_probs=y_prob_adapted, label_names=label_columns, threshold=0.5, verbose=False
    )
    
    return total_loss / len(dataloader), macro_f1, micro_f1, prec, rec, top_5_df, thresholds
