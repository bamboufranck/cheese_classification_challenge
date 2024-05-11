from transformers import BeitForImageClassification
import torch
import torch.nn as nn


class CustomBeitClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model_name='microsoft/beit-base-patch16-224-pt22k-ft22k', frozen=False):
        super().__init__()
        # Charger le modèle pré-entraîné
        self.beit = BeitForImageClassification.from_pretrained(pretrained_model_name)
        
        # Optionnellement, geler les paramètres du modèle pré-entraîné pour le fine-tuning
        if frozen:
            for param in self.beit.parameters():
                param.requires_grad = False
        
        # Modifier la dernière couche de classification
        self.beit.classifier = nn.Identity()  # Ignorer la tête de classification originale

        # Ajouter des couches supplémentaires pour la classification
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.beit.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # Passer l'input à travers le modèle pré-entraîné
        outputs = self.beit(x)
        x = outputs.logits  # Utiliser les logits comme nouvelles features
        
        # Passer les features à travers les nouvelles couches
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
