# ==========================================
# FILE: metrics.py
# ROLE: Multi-label Evaluation Metrics & Error Analysis
# ==========================================
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix
from typing import List, Tuple

def evaluate_goemotions(
    y_true: np.ndarray, 
    y_pred_probs: np.ndarray, 
    label_names: List[str], 
    threshold: float = 0.5
    verbose: bool = True
) -> Tuple[float, float, float, float, pd.DataFrame]:
    """
    Evaluates a Multi-label classification model for the GoEmotions dataset.
    Computes Macro/Micro F1, Precision, Recall, and generates a False Negative report.
    
    Args:
        y_true (np.ndarray): Ground truth multi-hot encoded matrix.
        y_pred_probs (np.ndarray): Predicted probabilities from Sigmoid.
        label_names (List[str]): List of the emotion class names.
        threshold (float): Probability threshold to convert to binary predictions.
        
    Returns:
        Tuple containing: macro_f1, micro_f1, precision, recall, top_5_hardest_df
    """
    # 1. Convert probabilities to binary predictions based on the threshold
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # 2. Calculate CORE METRICS: Macro F1, Micro F1, Precision, and Recall
    # Using 'micro' for Precision and Recall to match the overall report style
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # --- NEW: Added Precision and Recall calculations ---
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    # 3. Calculate Multi-label Confusion Matrix for ERROR ANALYSIS
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    fn_analysis = []
    
    for i, label in enumerate(label_names):
        tn, fp, fn, tp = mcm[i].ravel()
        actual_positives = fn + tp
        fn_rate = (fn / actual_positives) if actual_positives > 0 else 0.0
        
        fn_analysis.append({
            'Emotion': label,
            'FN Rate': f"{fn_rate * 100:.2f}%",
            'False Negatives (FN)': int(fn),
            'Actual Positives': int(actual_positives)
        })
        
    # 4. Process the Error DataFrame
    df_errors = pd.DataFrame(fn_analysis)
    df_errors['Sort_Key'] = df_errors['FN Rate'].str.rstrip('%').astype(float)
    df_errors = df_errors.sort_values(by=['Sort_Key', 'False Negatives (FN)'], ascending=[False, False])
    df_errors = df_errors.drop(columns=['Sort_Key'])
    
    top_5_hardest = df_errors.head(5)
    
    #5. If verbose = True -> print
    if verbose: 
        print("==================================================")
        print("OVERALL PERFORMANCE REPORT (CORE METRICS)") 
        print("==================================================")
        print(f"Macro F1-score : {macro_f1:.4f} (Primary Metric)")
        print(f"Micro F1-score : {micro_f1:.4f}")
        print(f"Precision      : {precision:.4f}")
        print(f"Recall         : {recall:.4f}\n")
        print("==================================================")
        print("ERROR ANALYSIS REPORT - FALSE NEGATIVES (FOR PHUONG)")
        print("==================================================")
        print("Description: Top 5 emotions the model is most likely to miss (Predicted 0, Actual 1).")
        print("Action item: Consider reviewing class_weights or augmenting data for these labels.\n")
        
        print(top_5_hardest.to_string(index=False))
        print("\n==================================================")
    
    # --- UPDATED: Return all 5 values ---
    return macro_f1, micro_f1, precision, recall, top_5_hardest 
