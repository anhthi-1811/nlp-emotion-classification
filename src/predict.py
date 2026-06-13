# =====================================================================
# FILE: predict.py
# ROLE: Independent prediction pipeline using structured checkpoints
# =====================================================================

import argparse
import os
import torch
import numpy as np
from transformers import DistilBertTokenizer, RobertaTokenizer
from models_bert import DistilBertEmotionClassifier, RoBERTaEmotionClassifier

def predict_emotion(text, experiment_type, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"-> Extracting serialized metrics from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    label_names = checkpoint["label_names"]
    thresholds = checkpoint["thresholds"]
    num_classes = len(label_names)

    if "roberta" in experiment_type:
        model = RoBERTaEmotionClassifier(num_classes=num_classes).to(device)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    else:
        model = DistilBertEmotionClassifier(num_classes=num_classes).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encodings = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0] 

    predicted_emotions = []
    print("\n--- EMOTION ANALYSIS GRAPH RESULTS ---")
    for i, emotion_name in enumerate(label_names):
        status = "PASSED" if probs[i] >= thresholds[i] else "FAILED"
        if probs[i] >= thresholds[i] or probs[i] > 0.3:
            print(f" > Sentiment [{emotion_name}]: Prob {probs[i]*100:.2f}% | Bound: {thresholds[i]:.2f} -> Outcome: {status}")
            
        if probs[i] >= thresholds[i]:
            predicted_emotions.append(emotion_name)

    if not predicted_emotions:
        predicted_emotions.append("neutral (unresolved boundaries)")

    return predicted_emotions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoEmotions Standalone Inference Gateway")
    parser.add_argument("--text", type=str, required=True, help="Input string for classification analysis.")
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True, 
        choices=[
            "distilbert_partial_freeze", 
            "distilbert_full_tuned_weighted", 
            "distilbert_full_tuned_unweighted", 
            "roberta_full_tuned"
        ]
    )
    args = parser.parse_args()

    ckpt_file = os.path.join("weights", f"best_{args.experiment}.pt")
    final_output = predict_emotion(args.text, args.experiment, ckpt_file)
    print(f"\n>>> FINAL CLASSIFIED SENTIMENT LABELS: {final_output}")
