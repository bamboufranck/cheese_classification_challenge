import torch
import torch.nn as nn

# Charger un modèle Vision Transformer pré-entraîné via torch.hub
class CustomViTClassifier(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        # Charger le modèle pré-entraîné (changer 'google/research/vision_transformer' par le repo approprié)
        self.vit = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

        # Optionnellement, geler les paramètres du modèle pré-entraîné pour le fine-tuning
        if frozen:
            for param in self.vit.parameters():
                param.requires_grad = False

            if unfreeze_last_layer:
                # Supposons que la dernière couche est accessible et que nous souhaitons la dégeler
                for param in self.vit.blocks[-1].parameters():
                    param.requires_grad = True

        # Ignorer la tête de classification originale et ajouter une nouvelle
        self.vit.head = nn.Identity()
        
        # Ajouter des couches supplémentaires pour la classification
        self.feature_extractor = nn.Sequential(
            nn.Linear(768, 1024),  # Assurez-vous que 768 est la taille correcte de la feature
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Passer l'input à travers le modèle pré-entraîné
        x = self.vit(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
