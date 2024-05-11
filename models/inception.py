import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms


class InceptionFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        # Charger un modèle Inception v3 pré-entraîné
        self.backbone = models.inception_v3(pretrained=True)
        # Ignorer la tête de classification originale
        self.backbone.fc = nn.Identity()

        if frozen:
            # Geler tous les paramètres du modèle
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Optionnellement dégeler la dernière couche BatchNorm
            if unfreeze_last_layer:
                # Dégeler les paramètres de la dernière couche BatchNorm avant la classification
                for param in self.backbone.Mixed_7c.parameters():
                    param.requires_grad = True

        # L'Inception v3 a une sortie de 2048 features avant la couche de classification
        self.classifier1 = nn.Linear(2048, 768)
        self.classifier2 = nn.Linear(768, num_classes)

    def forward(self, x):
        # Inception v3 requiert que l'entrée soit de taille (299, 299)
        # Assurez-vous que l'entrée x est déjà de cette taille avant de l'envoyer à ce modèle
        x = self.backbone(x)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x
