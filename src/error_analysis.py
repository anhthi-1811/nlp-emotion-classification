# =====================================================================
# FILE: error_analysis.py
# ROLE: Analyze weaknesses of the BiLSTM model (Multi-label, 28 emotions)
# =====================================================================

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, jaccard_score

from metrics import evaluate_goemotions
from data_pipeline import DataPipelineManager
from train_bilstm import BiLSTMClassifier, CustomSimpleTokenizer


# =====================================================================
# 1. LOAD CHECKPOINT + PREDICT ON TEST SET
# =====================================================================
def rebuild_tokenizer_from_vocab(vocab):
    """
    The checkpoint only saves 'vocab' (dict token->id), not the entire 
    CustomSimpleTokenizer object. This function creates a new instance 
    and assigns the saved vocab, so tokenizer.text_to_ids(text) can be 
    called just like during training.
    """
    tokenizer = CustomSimpleTokenizer(min_freq=2)
    tokenizer.vocab = vocab
    return tokenizer


def load_model_and_predict(checkpoint_path, test_loader, device):
    """
    Load checkpoint (.pt) containing model_state_dict + vocab + label_names + config,
    run predict on test_loader, return y_true, y_prob, label_names.
    """
    checkpoint  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab       = checkpoint["vocab"]
    label_names = checkpoint["label_names"]
    cfg         = checkpoint["config"]
    pad_idx     = vocab["<pad>"]
    unk_idx     = vocab["<unk>"]

    tokenizer = rebuild_tokenizer_from_vocab(vocab)

    model = BiLSTMClassifier(
        vocab_size    = cfg["vocab_size"],
        embedding_dim = cfg["embedding_dim"],
        hidden_dim    = cfg["hidden_dim"],
        output_dim    = len(label_names),
        n_layers      = cfg["n_layers"],
        bidirectional = True,
        dropout       = cfg["dropout"],
        pad_idx       = pad_idx
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            labels       = labels.to(device)
            tokenized    = [torch.tensor(tokenizer.text_to_ids(t)) for t in texts]
            tokenized    = [t if len(t) > 0 else torch.tensor([unk_idx]) for t in tokenized]
            input_ids    = nn.utils.rnn.pad_sequence(tokenized, batch_first=True, padding_value=pad_idx).to(device)
            text_lengths = torch.tensor([len(t) for t in tokenized])
            logits       = model(input_ids, text_lengths)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_prob = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    return y_true, y_prob, label_names


# =====================================================================
# 2. CONFUSION MATRIX (MULTI-LABEL) - PLOT FOR IMPORTANT LABELS
# =====================================================================
def plot_confusion_matrices(y_true, y_pred, label_names, labels_to_plot=None,
                             save_path="outputs/confusion_matrices.png"):
    """
    Plot multilabel confusion matrix as a 2x2 heatmap for each label in labels_to_plot.
    If labels_to_plot is not provided, take the top 8 labels with the most positives
    (so the plot isn't too cluttered with 28 subplots).
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    if labels_to_plot is None:
        positive_counts = y_true.sum(axis=0)
        top_idx = np.argsort(positive_counts)[::-1][:8]
        labels_to_plot = [label_names[i] for i in top_idx]

    n = len(labels_to_plot)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, label in enumerate(labels_to_plot):
        i  = label_names.index(label)
        cm = mcm[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[ax_idx],
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        axes[ax_idx].set_title(label)

    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[+] Saved confusion matrices -> {save_path}")
    plt.close()


# =====================================================================
# 3. FILTER COMPLETELY WRONG PREDICTIONS (10-15 SENTENCES)
# =====================================================================
def find_completely_wrong_samples(df_test, y_true, y_pred, label_names,
                                   n_samples=15, seed=42):
    """
    "Completely wrong" = Jaccard(y_true_row, y_pred_row) == 0
    (no common labels between prediction and ground-truth; ignore cases
    where both are empty as there is no information to analyze).
    """
    rows = []
    for idx in range(len(y_true)):
        true_set = set(np.where(y_true[idx] == 1)[0])
        pred_set = set(np.where(y_pred[idx] == 1)[0])

        if len(true_set) == 0 and len(pred_set) == 0:
            continue

        jacc = jaccard_score(y_true[idx], y_pred[idx], zero_division=0)
        if jacc == 0:
            rows.append({
                "idx": idx,
                "text": df_test.iloc[idx]["text"],
                "true_labels": ", ".join(label_names[i] for i in true_set) or "(none)",
                "pred_labels": ", ".join(label_names[i] for i in pred_set) or "(none)",
                "reason": ""  # leave blank, fill in manually after reading the sentence
            })

    df_wrong = pd.DataFrame(rows)
    print(f"[INFO] Total completely wrong predictions (Jaccard=0): {len(df_wrong)}")

    if len(df_wrong) > n_samples:
        df_wrong = df_wrong.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    return df_wrong


# =====================================================================
# 4. GROUP 27 LABELS -> 4 CLUSTERS (FOLLOWING GOOGLE GOEMOTIONS STANDARD)
# Sentiment grouping (see GoEmotions paper / repo - 'neutral' separated)
# =====================================================================
SENTIMENT_GROUPS = {
    "positive": [
        "admiration", "amusement", "approval", "caring", "desire",
        "excitement", "gratitude", "joy", "love", "optimism",
        "pride", "relief"
    ],
    "negative": [
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"
    ],
    "ambiguous": [
        "confusion", "curiosity", "realization", "surprise"
    ],
    "neutral": [
        "neutral"
    ]
}


def label_to_group(label_names):
    """Return dict {label_name: group_name}, warn if missing mapping."""
    mapping = {}
    for group, labels in SENTIMENT_GROUPS.items():
        for lb in labels:
            mapping[lb] = group
    missing = [lb for lb in label_names if lb not in mapping]
    if missing:
        print(f"[WARN] Labels not mapped to any group: {missing}")
    return mapping


def plot_group_confusion(y_true, y_pred, label_names,
                          save_path="outputs/group_confusion.png"):
    """
    4x4 Confusion matrix at the CLUSTER level (group-level).
    For each sample, the "dominant group" = the group with the most active labels
    (true and pred calculated separately). If no label is active -> 'neutral'.
    """
    mapping    = label_to_group(label_names)
    groups     = ["positive", "negative", "ambiguous", "neutral"]
    group_idx  = {g: [i for i, lb in enumerate(label_names) if mapping.get(lb) == g] for g in groups}

    def dominant_group(row):
        counts = {g: row[idx].sum() for g, idx in group_idx.items()}
        if sum(counts.values()) == 0:
            return "neutral"
        return max(counts, key=counts.get)

    true_groups = [dominant_group(y_true[i]) for i in range(len(y_true))]
    pred_groups = [dominant_group(y_pred[i]) for i in range(len(y_pred))]

    cm = pd.crosstab(pd.Series(true_groups, name="True Group"),
                      pd.Series(pred_groups, name="Pred Group"))
    cm = cm.reindex(index=groups, columns=groups, fill_value=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
    plt.title("Confusion Matrix - 4 Sentiment Groups (Google GoEmotions)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[+] Saved group-level confusion matrix -> {save_path}")
    plt.close()

    return cm


# =====================================================================
# 5. DETECT "ILLOGICAL" PREDICTIONS
# (e.g., anger + amusement, sadness + joy assigned to 1 sentence)
# =====================================================================
CONTRADICTORY_PAIRS = [
    ("anger", "amusement"),
    ("anger", "joy"),
    ("sadness", "joy"),
    ("sadness", "amusement"),
    ("disgust", "admiration"),
    ("fear", "excitement"),
    ("annoyance", "approval"),
    ("disapproval", "approval"),
    ("grief", "joy"),
    ("embarrassment", "pride"),
]


def find_contradictory_predictions(df_test, y_pred, label_names, n_examples=3):
    """
    For each "illogical" label pair, find samples where the model predicts BOTH
    labels simultaneously. Return DataFrame {pair, count, examples}.
    """
    results = []
    label_to_idx = {lb: i for i, lb in enumerate(label_names)}

    for label_a, label_b in CONTRADICTORY_PAIRS:
        if label_a not in label_to_idx or label_b not in label_to_idx:
            continue
        idx_a, idx_b = label_to_idx[label_a], label_to_idx[label_b]
        mask  = (y_pred[:, idx_a] == 1) & (y_pred[:, idx_b] == 1)
        count = int(mask.sum())

        examples = []
        if count > 0:
            sample_indices = np.where(mask)[0][:n_examples]
            examples = [df_test.iloc[i]["text"] for i in sample_indices]

        results.append({
            "pair": f"{label_a} + {label_b}",
            "count": count,
            "examples": " || ".join(examples)
        })

    return pd.DataFrame(results).sort_values("count", ascending=False)


# =====================================================================
# 6. ANALYZE FALSE NEGATIVES FOR LABELS WITH LITTLE DATA
# =====================================================================
def analyze_rare_label_fn(y_true, y_pred_probs, label_names, threshold=0.55,
                           rare_threshold=200):
    """
    Call evaluate_goemotions() to get overall metrics, then manually recalculate
    full FNs (28 labels) and filter only labels with "Actual Positives"
    < rare_threshold (rare labels).
    """
    macro_f1, micro_f1, precision, recall, _ = evaluate_goemotions(
        y_true=y_true, y_pred_probs=y_pred_probs,
        label_names=label_names, threshold=threshold, verbose=False
    )

    y_pred = (y_pred_probs >= threshold).astype(int)
    mcm    = multilabel_confusion_matrix(y_true, y_pred)

    rows = []
    for i, label in enumerate(label_names):
        tn, fp, fn, tp = mcm[i].ravel()
        actual_pos = fn + tp
        fn_rate    = (fn / actual_pos * 100) if actual_pos > 0 else 0
        rows.append({
            "Emotion": label,
            "Actual Positives": int(actual_pos),
            "False Negatives": int(fn),
            "FN Rate (%)": round(fn_rate, 2),
            "Is Rare": actual_pos < rare_threshold
        })

    df_all  = pd.DataFrame(rows).sort_values("FN Rate (%)", ascending=False)
    df_rare = df_all[df_all["Is Rare"]].copy()

    return {
        "macro_f1": macro_f1, "micro_f1": micro_f1,
        "precision": precision, "recall": recall,
        "fn_table_all": df_all,
        "fn_table_rare": df_rare,
    }


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running error analysis on device: {device}")

    os.makedirs("outputs", exist_ok=True)

    # --- 0. Load test data ---
    TEST_CSV    = "data/processed/test.csv"
    df_test     = pd.read_csv(TEST_CSV, encoding='utf-8-sig')
    LABEL_NAMES = [col for col in df_test.columns if col not in ['text', 'id']]

    pipeline    = DataPipelineManager(raw_data_path=TEST_CSV, mock_data_output_path=TEST_CSV)
    test_loader = pipeline.get_pytorch_loaders(df_test, batch_size=64, shuffle=False)

    # --- 1. Load checkpoint + predict ---
    # Rename checkpoint file to match the one saved in weights/
    CHECKPOINT_PATH = "weights/best_bilstm_BiLSTM_w+GloVe+HD512+Ep30+Sched+Clip.pt"
    y_true, y_prob, label_names = load_model_and_predict(CHECKPOINT_PATH, test_loader, device)

    # Take from previous tune_threshold() results (Exp3 -> best threshold = 0.55)
    BEST_THRESHOLD = 0.55
    y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2: CONFUSION MATRIX FOR POPULAR LABELS")
    print("="*60)
    plot_confusion_matrices(y_true, y_pred, label_names,
                             save_path="outputs/confusion_matrices.png")

    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 3: FILTER COMPLETELY WRONG PREDICTIONS")
    print("="*60)
    df_wrong = find_completely_wrong_samples(df_test, y_true, y_pred, label_names, n_samples=15)
    df_wrong.to_csv("outputs/wrong_predictions.csv", index=False)
    print(df_wrong[["text", "true_labels", "pred_labels"]].to_string(index=False))
    print("[+] Saved -> outputs/wrong_predictions.csv")
    print("[TODO] Open this CSV file, read each sentence and fill in the 'reason' column:")
    print("       - ambiguous words, hard to clearly name the emotion")
    print("       - ground truth labels might be misleading (annotator disagreement)")
    print("       - lack of context (sarcasm, need to know previous context)")

    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: GROUP 27 LABELS -> 4 CLUSTERS (GOOGLE GOEMOTIONS)")
    print("="*60)
    group_cm = plot_group_confusion(y_true, y_pred, label_names,
                                     save_path="outputs/group_confusion.png")
    print(group_cm)

    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5: ILLOGICAL PREDICTIONS")
    print("="*60)
    df_contra = find_contradictory_predictions(df_test, y_pred, label_names, n_examples=3)
    print(df_contra.to_string(index=False))
    df_contra.to_csv("outputs/contradictory_predictions.csv", index=False)
    print("[+] Saved -> outputs/contradictory_predictions.csv")

    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 6: FALSE NEGATIVES FOR LABELS WITH LITTLE DATA")
    print("="*60)
    fn_results = analyze_rare_label_fn(y_true, y_prob, label_names,
                                        threshold=BEST_THRESHOLD, rare_threshold=200)
    print(f"Test Macro F1: {fn_results['macro_f1']:.4f}")
    print("\n-- RARE labels (Actual Positives < 200) --")
    print(fn_results["fn_table_rare"].to_string(index=False))
    fn_results["fn_table_all"].to_csv("outputs/fn_table_all_labels.csv", index=False)
    print("[+] Saved -> outputs/fn_table_all_labels.csv")

    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("DONE. Output files are located in outputs/:")
    print("  - confusion_matrices.png       (Step 2)")
    print("  - wrong_predictions.csv        (Step 3 - fill in 'reason' column)")
    print("  - group_confusion.png          (Step 4)")
    print("  - contradictory_predictions.csv (Step 5)")
    print("  - fn_table_all_labels.csv      (Step 6)")
    print("="*60)