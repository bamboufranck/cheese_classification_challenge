import torch
import torch.nn as nn
from transformers import BeitForImageClassification



class CustomBeitClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model_name='microsoft/beit-base-patch16-224-pt22k-ft22k', frozen=False, unfreeze_last_layer=True):
        super().__init__()
        # Charger le modèle Beit pré-entraîné
        self.beit = BeitForImageClassification.from_pretrained(pretrained_model_name)

        # Optionnellement, geler les paramètres du modèle pré-entraîné
        if frozen:
            for param in self.beit.parameters():
                param.requires_grad = False

            # Si spécifié, dégeler la dernière couche
            if unfreeze_last_layer:
                for param in self.beit.encoder.layer[-1].parameters():
                    param.requires_grad = True

        # Remplacer la tête de classification originale par une identité
        self.beit.classifier = nn.Identity()

        # Ajouter de nouvelles couches pour la classification finale
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.beit.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Extraire les logits à partir du modèle Beit pré-entraîné
        outputs = self.beit(x)
        x = outputs.logits  # Extraire les logits

        # Passer ces logits à travers les nouvelles couches de classification
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
