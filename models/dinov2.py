import torch
import torch.nn as nn
from transformers import DeiTFeatureExtractor, DeiTModel
from transformers import ViTImageProcessor, ViTModel


class DinoV2Finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        #self.backbone = ViTModel.from_pretrained('google/vit-large-patch16-224')
        #self.backbone = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        #self.backbone= torch.hub.load('google/vit-base-patch16-224-in21k', 'vit_large_patch16_224', pretrained=True)
        self.backbone.head = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            if unfreeze_last_layer:
    
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
                for param in self.backbone.blocks[-1].parameters():
                    param.requires_grad = True
            

        self.features_dim = self.backbone.num_features
        #self.dropout = nn.Dropout(0.5)
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
    
"""""""""


import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np

class DinoV2Finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        
        # Load the backbone (ViT Model)
        self.processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
        self.backbone = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
        
        # Remove the classification head
        self.backbone.pooler = nn.Identity()
        
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            if unfreeze_last_layer:
                for param in self.backbone.layernorm.parameters():
                    param.requires_grad = True
                for param in self.backbone.encoder.layer[-1].parameters():
                    param.requires_grad = True
        
        self.features_dim = self.backbone.config.hidden_size
        self.classifier = nn.Linear(self.features_dim, num_classes)

    def forward(self, images):
        # Preprocess images
        inputs = self.processor(images=images, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.backbone.device) for k, v in inputs.items()}
        
        # Get the outputs from the backbone model
        outputs = self.backbone(**inputs)
        
        # Use the CLS token for classification (assuming the first token is the CLS token)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(cls_token)
        return logits

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image

"""""""""