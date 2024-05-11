from transformers import  ViTForImageClassification
import torch.nn as nn

class GoogleVitFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super().__init__()
        self.backbone =  ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.backbone.fc = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier1=nn.Linear(1000, 768)
        self.classifier2 = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs= self.backbone(x)
        x = outputs.logits
        x = self.classifier1(x)
        x= self.classifier2(x)
        return x