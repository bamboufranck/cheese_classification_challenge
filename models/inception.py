import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms


class InceptionFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        # Charger un modèle Inception v3 pré-entraîné
        self.backbone = models.inception_v3(pretrained=True)
        # Ignorer la tête de classification originale
        self.backbone.fc = nn.Identity()

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # L'Inception v3 a une sortie de 2048 features avant la couche de classification
        self.classifier1 = nn.Linear(2048, 768)
        self.classifier2 = nn.Linear(768, num_classes)

    def forward(self, x):
        # Inception v3 requiert que l'entrée soit de taille (299, 299)
        transform = transforms.Compose([
    transforms.Resize((299, 299))])
        x=transform(x)
        x = self.backbone(x)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x
