import torch
import torch.nn as nn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BertModel, BertTokenizer

from transformers import DeiTFeatureExtractor, DeiTModel
from transformers import ViTFeatureExtractor, ViTModel


def denormalize(tensor):
    # Convertit un tenseur normalisé (ImageNet) en un tenseur avec des valeurs entre 0 et 1
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()

    tensor = tensor * std + mean  # Appliquer l'inverse de la normalisation
    tensor = torch.clamp(tensor, 0.0, 1.0)  # Clamp les valeurs pour s'assurer qu'elles sont entre 0 et 1
    return tensor




class DinoV2Finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        #self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone = ViTModel.from_pretrained('google/vit-large-patch16-224')
        #self.backbone = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        #self.backbone= torch.hub.load('google/vit-base-patch16-224-in21k', 'vit_large_patch16_224', pretrained=True)
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                #for param in self.backbone.norm.parameters():
                    #param.requires_grad = True
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True


        self.features_dim = self.backbone.num_features
        #self.dropout = nn.Dropout(0.7)
        #self.batch_norm = nn.BatchNorm1d(self.features_dim)
        #self.activation = nn.ReLU()
        self.classifier= nn.Linear(self.features_dim, num_classes)


    def forward(self, x):
        x= self.backbone(x)
        #x = self.dropout(x)
        #x = self.batch_norm(x)
        x = self.classifier(x)
        #x = self.activation(x)
        #x = self.classifier1(x)

    
        return x
    







"""""
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
        self.classifier = nn.Linear(self.backbone.norm.normalized_shape[0], num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    
"""""
