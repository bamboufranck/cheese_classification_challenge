import torch
from torch import nn
from timm import create_model

class EfficientNetFinetune(nn.Module):
    def __init__(self, num_classes, model_name='swin_small_patch4_window7_224', pretrained=True):
        super(SwinTransformerClassifier, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)