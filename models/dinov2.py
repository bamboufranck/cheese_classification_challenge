import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import DeiTFeatureExtractor, DeiTModel
from transformers import ViTFeatureExtractor, ViTModel


"""""

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
    

"""""





class DinoV2Finetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = ViTModel.from_pretrained('google/vit-large-patch16-224')
        self.backbone.pooler = nn.Identity()  # Set the pooler to identity if it exists

            # Freeze all layers if required
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
                # Unfreeze the last layer if required
            if unfreeze_last_layer:
                for param in self.backbone.encoder.layer[-1].parameters():
                    param.requires_grad = True

            # Determine the number of features based on the backbone
        num_features = self.backbone.config.hidden_size

            # Define the classifier
        self.classifier = nn.Sequential(
                nn.Linear(num_features, num_classes)
            )

    def forward(self, x):
        # Extract features using the backbone
        outputs = self.backbone(x).last_hidden_state
        # Pool the outputs (take the mean of the sequence)
        pooled_output = outputs.mean(dim=1)
        # Pass through the classifier
        logits = self.classifier(pooled_output)
        return logits





