import torch
import torch.nn as nn


class DinoV2Finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        #self.backbone= torch.hub.load('google/vit-base-patch16-224-in21k', 'vit_large_patch16_224', pretrained=True)
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.backbone.blocks[-4:].parameters():
                    param.requires_grad = True
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True

        self.features_dim = self.backbone.num_features
        #self.dropout = nn.Dropout(0.7)
        #self.batch_norm = nn.BatchNorm1d(self.features_dim)
        self.activation = nn.ReLU()
        #self.classifier= nn.Linear(self.features_dim, num_classes)
        self.classifier = nn.Linear(self.features_dim, 768)
        #self.activation = nn.ReLU()
        self.classifier1 = nn.Linear(768, num_classes)
       

    def forward(self, x):
        x = self.backbone(x)
        #x = self.dropout(x)
        #x = self.batch_norm(x)
        x = self.classifier(x)
        #x = self.activation(x)
        x = self.classifier1(x)
    
        return x
