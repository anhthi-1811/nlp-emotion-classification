# Multi-Label Emotion Classification on the GoEmotions Dataset

This project investigates and compares deep learning approaches for **multi-label emotion classification** using the GoEmotions dataset.

Implemented models include:

* BiLSTM
* BiLSTM + GloVe
* DistilBERT
* RoBERTa

The main objective is to compare traditional recurrent neural networks with Transformer-based language models and evaluate the impact of class imbalance handling using weighted BCE loss (`pos_weight`).

---

## Objectives

* Predict multiple emotion labels from a single text input.
* Compare BiLSTM-based models with Transformer architectures.
* Evaluate the effectiveness of weighted BCE loss for rare emotion classes.
* Optimize classification thresholds to improve F1-score.

---

## Project Structure

```text
PROJECT/
├── data/
│   ├── raw/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── notebooks/
│   ├── 01_Data_Pipeline_and_EDA.ipynb
│   └── 02_Text_Preprocessing.ipynb
│
├── results/
│
├── src/
│   ├── data_pipeline.py
│   ├── engine.py
│   ├── error_analysis.py
│   ├── eval_test.py
│   ├── metrics.py
│   ├── models_bert.py
│   ├── predict.py
│   ├── preprocessor.py
│   ├── main.py
│   ├── train_bilstm.py
│   └── utils.py
│
├── weights/
├── .gitignore
└── requirements.txt
```

---

## Dataset

The project uses the GoEmotions dataset, which contains Reddit comments annotated with 28 emotion categories.

Available datasets:

* `train.csv`
* `val.csv`
* `test.csv`

All datasets have been preprocessed and are ready for training.

### Data Pipeline

* Transformers use Hugging Face tokenizers.
* BiLSTM uses a custom tokenizer and vocabulary.
* Data is loaded through PyTorch Dataset and DataLoader.

---

## Experimental Setup

### Transformer Models

| Experiment                       | Description                             |
| -------------------------------- | --------------------------------------- |
| distilbert_partial_freeze        | Partial fine-tuning of DistilBERT       |
| distilbert_full_tuned_weighted   | Full fine-tuning with weighted BCE loss |
| distilbert_full_tuned_unweighted | Full fine-tuning with standard BCE loss |
| roberta_full_tuned               | Full fine-tuning of RoBERTa-base        |

### BiLSTM Models

| Experiment | Description                                             |
| ---------- | ------------------------------------------------------- |
| Exp 1      | Baseline BiLSTM                                         |
| Exp 2      | BiLSTM + GloVe                                          |
| Exp 3      | Extended architecture with Scheduler and Early Stopping |
| Exp 4      | Exp 3 without class weights                             |
| Exp 5      | High-regularization configuration                       |

---

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Login to Weights & Biases:

```bash
wandb login
```

---

## Training

### Transformer

```bash
python src/train_bert.py --experiment distilbert_full_tuned_weighted
```

Change the `--experiment` argument to run different DistilBERT or RoBERTa configurations.

### BiLSTM

```bash
python src/train_bilstm.py
```

---

## Evaluation

Evaluate a trained model on the test set:

```bash
python src/eval_test.py
```

Evaluation metrics:

* Macro F1
* Micro F1
* Precision
* Recall

---

## Inference

Predict emotions for a custom input text:

```bash
python src/predict.py --text "Your input text" --experiment <experiment_name>
```

Example output:

```text
-> Extracting serialized metrics from: weights/<tên_checkpoint>.pt...

--- EMOTION ANALYSIS GRAPH RESULTS ---
 > Sentiment [admiration]: Prob 85.42% | Bound: 0.42 -> Outcome: PASSED
 > Sentiment [joy]: Prob 64.15% | Bound: 0.38 -> Outcome: PASSED
 > Sentiment [love]: Prob 32.40% | Bound: 0.45 -> Outcome: FAILED

[+] Input Text: "Câu văn bản bạn muốn kiểm tra"
[+] Predicted Emotion Labels: ['admiration', 'joy']
```

---

## Main Features

* Support for BiLSTM, DistilBERT, and RoBERTa.
* Weighted BCE Loss for imbalanced labels.
* Automatic threshold tuning.
* Early stopping and learning rate scheduling.
* Error analysis for difficult samples.
* Training visualization using Weights & Biases.

---

## Results

Below is the consolidated performance summary evaluated across both Deep Learning and Transformer paradigms.

### Transformer Models

| ID | Model      | Training Strategy | Loss Function  | Val Loss | Macro F1    | Micro F1    |
| -- | ---------- | ----------------- | -------------- | -------- | ----------- | ----------- |
| M1 | DistilBERT | Partial Freeze    | Weighted BCE   | 0.79779  | 0.50637     | 0.56495     |
| M2 | DistilBERT | Full Tuned        | Unweighted BCE | 0.20408  | **0.51395** | **0.57154** |
| M3 | DistilBERT | Full Tuned        | Weighted BCE   | 0.92900  | 0.50584     | 0.56426     |
| M4 | RoBERTa    | Full Tuned        | Weighted BCE   | 0.79485  | 0.51394     | 0.56786     |

### BiLSTM Models

| Exp | Configuration       | Epoch | Train Loss | Val Loss | Macro F1   | Micro F1   |
| --- | ------------------- | ----- | ---------- | -------- | ---------- | ---------- |
| 1   | Baseline            | 10/10 | 0.8691     | 0.9133   | 0.3633     | 0.4083     |
| 2   | + Frozen GloVe      | 9/10  | 0.7904     | 0.8801   | 0.3881     | 0.4422     |
| 3   | Advanced            | 30/30 | 0.4655     | 1.5083   | 0.4283     | 0.4818     |
| 4   | Advanced Unweighted | 23/30 | 0.1473     | 0.2402   | **0.4419** | **0.5265** |
| 5   | High Regularization | 12/30 | 0.8288     | 0.8741   | 0.3804     | 0.4259     |

### Key Findings

* DistilBERT Full Fine-Tuning (M2) achieved the best overall performance.
* RoBERTa provided almost identical performance despite having more parameters.
* Weighted BCE Loss did not significantly improve Transformer performance.
* GloVe embeddings improved BiLSTM over the baseline model.
* Transformer-based models outperformed BiLSTM by approximately 7–8 Macro F1 points.
