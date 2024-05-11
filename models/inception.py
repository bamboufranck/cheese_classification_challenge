import torchvision.models as models
import torch.nn as nn

class CustomMobileNetFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier[1] = nn.Identity()

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.backbone.features[-1].parameters():  # Last block in MobileNetV2
                    param.requires_grad = True
        
        num_features = self.backbone.classifier[1].in_features
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
