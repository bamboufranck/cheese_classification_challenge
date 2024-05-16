import torch
from torch import nn
from timm import create_model
import torch.nn as nn
from timm import create_model  # Assurez-vous d'avoir installé la bibliothèque `timm`

class EfficientNetFinetune(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetFinetune, self).__init__()  # Correction ici
        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
