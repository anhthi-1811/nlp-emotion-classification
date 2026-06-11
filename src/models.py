# =====================================================================
# FILE: models.py
# ROLE: Model architecture definitions for GoEmotions classification
# =====================================================================

import torch
import torch.nn as nn
from transformers import DistilBertModel, RobertaModel

class DistilBertEmotionClassifier(nn.Module):
    """
    DistilBERT architecture for multi-label emotion classification
    supporting both full fine-tuning and progressive layer freezing.
    """
    def __init__(self, num_classes, freeze_backbone=False, partial_freeze=False):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        if freeze_backbone:
            if partial_freeze:
                # Freeze all layers except the top transformer blocks (Layers 4 & 5)
                for name, param in self.distilbert.named_parameters():
                    if "transformer.layer.4" in name or "transformer.layer.5" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                # Strict freeze of the entire backbone network
                for param in self.distilbert.parameters():
                    param.requires_grad = False
        else:
            print(">>> System: Initializing 100% full backbone training mode.")

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


class RoBERTaEmotionClassifier(nn.Module):
    """
    RoBERTa-base architecture for deep multi-label emotion classification.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])
