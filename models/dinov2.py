import torch
import torch.nn as nn


class DinoV2Finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True

        self.dropout = nn.Dropout(0.5)
        self.classifier1 = nn.Linear(self.backbone.norm.normalized_shape[0], 128)
        self.classifier2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier1(x)
        x = self.dropout(x)
        x = self.classifier2(x)
        return x
