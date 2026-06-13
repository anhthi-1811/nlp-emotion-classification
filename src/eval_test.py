# =====================================================================
# FILE: eval_test.py
# ROLE: Comprehensive evaluation across 4 experiments & best model selection
# =====================================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, RobertaTokenizer

# Reuse the core resources from your existing source modules
from data_pipeline import DataPipelineManager
from models_bert import DistilBertEmotionClassifier, RoBERTaEmotionClassifier
from sklearn.metrics import f1_score, classification_report

def evaluate_and_extract_errors(model, test_dataloader, df_test, checkpoint_path, device="cuda"):
    """
    Loads checkpoint, computes system loss, evaluates per-label performance,
    and extracts the top 5 critical error samples with the highest loss values.
    """
    print(f"\n--- Configuring Checkpoint: {checkpoint_path} ---")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimal_thresholds = checkpoint["thresholds"]
    label_names = checkpoint["label_names"]

    model.to(device)
    model.eval()

    all_raw_logits = []

    # 1. Execute Inference Pipeline
    with torch.no_grad():
        for texts, _ in test_dataloader:
            # Tokenize directly from raw text to ensure synchronization with original setup
            encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            all_raw_logits.append(logits.cpu().numpy())

    raw_logits = np.vstack(all_raw_logits)
    probabilities = 1 / (1 + np.exp(-raw_logits)) # Apply Sigmoid to compute probabilities

    # Extract ground truth matrix safely from the original DataFrame
    ground_truths = df_test[label_names].values

    # 2. Apply Dynamic Optimal Thresholds Per Class
    predictions = np.zeros_like(probabilities)
    for class_idx in range(len(label_names)):
        predictions[:, class_idx] = (probabilities[:, class_idx] >= optimal_thresholds[class_idx]).astype(int)

    # 3. Compute Binary Cross-Entropy (BCE) Loss per Sample (Sample-level Loss)
    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1 - eps)
    bce_per_sample_per_class = -(ground_truths * np.log(probabilities) + (1 - ground_truths) * np.log(1 - probabilities))
    sample_losses = np.mean(bce_per_sample_per_class, axis=1)
    average_test_loss = np.mean(sample_losses)

    # 4. Compute Global Core Metrics
    global_macro_f1 = f1_score(ground_truths, predictions, average='macro', zero_division=0)
    global_micro_f1 = f1_score(ground_truths, predictions, average='micro', zero_division=0)

    # 5. Generate Detailed Per-Label Performance Report
    test_report = classification_report(
        ground_truths, predictions, target_names=label_names, output_dict=True, zero_division=0
    )
    df_performance = pd.DataFrame(test_report).transpose().iloc[:-4] # Exclude final summary metrics rows

    # 6. Extract Top 5 Hardest Samples (Highest BCE Loss Values)
    hardest_indices = np.argsort(sample_losses)[::-1][:5]
    error_records = []

    for idx in hardest_indices:
        gt_labels = [label_names[i] for i, val in enumerate(ground_truths[idx]) if val == 1]
        pred_labels = [label_names[i] for i, val in enumerate(predictions[idx]) if val == 1]
        raw_text = df_test.iloc[idx]['text']

        error_records.append({
            "Sample_ID": f"#{idx}",
            "Text_Input": raw_text,
            "Ground_Truth": ", ".join(gt_labels) if gt_labels else "neutral",
            "Predicted": ", ".join(pred_labels) if pred_labels else "neutral",
            "Loss_Value": round(float(sample_losses[idx]), 4)
        })
    df_errors = pd.DataFrame(error_records)

    return df_performance, df_errors, {"test_loss": average_test_loss, "macro_f1": global_macro_f1, "micro_f1": global_micro_f1}


def plot_best_model_performance(df_perf, model_name):
    """
    Generates and saves a high-quality academic visualization chart combining
    Bar plot (F1-score) and Scatter plot (Precision & Recall) for the best model.
    """
    # Prepare and sort data by F1-score in descending order
    df_plot = df_perf.copy().reset_index().rename(columns={'index': 'Emotion'})
    df_plot = df_plot.sort_values(by='f1-score', ascending=False).reset_index(drop=True)

    # Configure chart styling suitable for academic thesis or technical papers
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 11))

    # Render horizontal bar plot for F1-score values
    sns.barplot(
        x='f1-score',
        y='Emotion',
        data=df_plot,
        palette='Blues_r',
        ax=ax
    )

    # Overlay scatter points for individual Precision and Recall values
    ax.scatter(df_plot['precision'], df_plot['Emotion'], color='#e74c3c', label='Precision', zorder=3, edgecolors='black', s=60)
    ax.scatter(df_plot['recall'], df_plot['Emotion'], color='#2ecc71', label='Recall', zorder=3, edgecolors='black', s=60)

    # Configure academic axis labels and titles
    ax.set_title(f'Per-Label Classification Performance ({model_name.upper()})\nSorted by F1-score', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Metric Value', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Emotion Labels', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlim(0.0, 1.0)

    # Annotate exact F1-score numerical values beside each bar
    for i, v in enumerate(df_plot['f1-score']):
        ax.text(v + 0.01, i + 0.15, f"{v:.4f}", color='black', fontweight='semibold', fontsize=10)

    # Attach legend labels to the lower right section
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=12)

    plt.tight_layout()

    # Export the visualization plot to the results directory
    output_image_path = f'results/performance_chart_{model_name}.png'
    plt.savefig(output_image_path, dpi=300)
    print(f"[SUCCESS] High-resolution evaluation chart successfully saved to: '{output_image_path}'")
    plt.show()


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DATA_PATH = os.path.join("data", "test.csv")
    NUM_CLASSES = 28

    # Define the list of the 4 experimental setups to evaluate
    experiments = [
        "distilbert_partial_freeze",
        "distilbert_full_tuned_weighted",
        "distilbert_full_tuned_unweighted",
        "roberta_full_tuned"
    ]

    # Verify the existence of the test dataset file
    if not os.path.exists(TEST_DATA_PATH):
        print(f"[ERROR] Test data file not found at: '{TEST_DATA_PATH}'")
        exit()

    df_test = pd.read_csv(TEST_DATA_PATH)

    # Instantiate DataPipelineManager to establish standard DataLoaders
    pipeline = DataPipelineManager(raw_data_path=TEST_DATA_PATH, mock_data_output_path=os.path.join("data", "mock_test.csv"))
    test_loader = pipeline.get_pytorch_loaders(df_test, batch_size=16, shuffle=False)

    summary_results = []
    best_macro_f1 = -1.0
    best_experiment_name = ""
    best_df_performance = None
    best_df_errors = None

    print(f"=== SYSTEM ACTIVATION: Benchmarking {len(experiments)} Experimental Model Architectures ===")

    # Iterate through all available checkpoints
    for exp in experiments:
        checkpoint_file = os.path.join("weights", f"best_{exp}.pt")
        if not os.path.exists(checkpoint_file):
            print(f"[WARNING] Skipping experiment '{exp}' as its checkpoint file does not exist.")
            continue

        # Dynamically instantiate the correct model architecture and matching tokenizer
        if "roberta" in exp:
            model = RoBERTaEmotionClassifier(num_classes=NUM_CLASSES)
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            model = DistilBertEmotionClassifier(num_classes=NUM_CLASSES)
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Run evaluation and analysis execution
        df_perf, df_err, metrics = evaluate_and_extract_errors(
            model=model, test_dataloader=test_loader, df_test=df_test, checkpoint_path=checkpoint_file, device=DEVICE
        )

        # Append compact performance records for comparative summary table
        summary_results.append({
            "Experiment": exp,
            "Test Loss": round(metrics["test_loss"], 4),
            "Macro F1": round(metrics["macro_f1"], 4),
            "Micro F1": round(metrics["micro_f1"], 4)
        })

        # Selection logic to track and isolate the best model setup based on Macro F1
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_experiment_name = exp
            best_df_performance = df_perf
            best_df_errors = df_err

    # =====================================================================
    # COMPREHENSIVE BENCHMARKING REPORT TERMINAL OUTPUT
    # =====================================================================
    df_summary = pd.DataFrame(summary_results)
    print("\n" + "="*80)
    print(" GLOBAL MODEL PERFORMANCE COMPARISON REPORT (TEST SET) ")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)

    if best_experiment_name:
        print(f"\nBEST PERFORMING MODEL ARCHITECTURE: {best_experiment_name.upper()}")
        print(f"Peak Global Macro F1-Score Achieved: {best_macro_f1:.4f}")

        print(f"\n PER-LABEL METRICS ANALYSIS REPORT FOR BEST MODEL")
        print("-" * 80)
        print(best_df_performance[['precision', 'recall', 'f1-score']].sort_values(by='f1-score', ascending=False).to_string())
        print("-" * 80)

        print(f"\n QUALITATIVE ERROR ANALYSIS: TOP 5 CRITICAL SAMPLES FOR BEST MODEL")
        print("-" * 80)
        for _, row in best_df_errors.iterrows():
            print(f"Sample ID: {row['Sample_ID']} | Maximum Sample BCE Loss: {row['Loss_Value']}")
            print(f"Raw Input Text: \"{row['Text_Input']}\"")
            print(f"Ground Truth (GT): [{row['Ground_Truth']}] <---> System Prediction: [{row['Predicted']}]")
            print("." * 60)
        print("="*80)

        # Serialize performance data to CSV outputs
        os.makedirs("results", exist_ok=True)
        df_summary.to_csv("results/model_comparison_test.csv", index=False)
        best_df_performance.to_csv(f"results/best_model_per_label_{best_experiment_name}.csv")
        print("[SUCCESS] Analytical metrics successfully exported to the 'results/' directory.")

        # Trigger automatic chart generation pipeline for the best performing model
        plot_best_model_performance(best_df_performance, best_experiment_name)