import torch 
import torch.nn as nn
import timm

class VisionTransformerFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        #self.classifier = nn.Linear(self.backbone.norm.normalized_shape[0], num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
