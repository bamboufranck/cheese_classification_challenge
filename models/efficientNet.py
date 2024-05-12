import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetFinetune(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', frozen=False, unfreeze_last_layer=True):
        super(EfficientNetFinetune, self).__init__()
        # Charger un modèle EfficientNet pré-entraîné
        self.backbone = models.efficientnet_b0(pretrained=True) if model_name == 'efficientnet_b0' else models.efficientnet_b1(pretrained=True)
        
        # Remplacer le classificateur pour le nombre de classes spécifié
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        
        # Geler les paramètres si nécessaire
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Dégeler la dernière couche si spécifié
        if unfreeze_last_layer:
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        # Faites passer les entrées à travers le backbone puis le classificateur
        return self.backbone(x)
