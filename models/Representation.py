import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel



class Encoder(nn.Module):
    def __init__(self,num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        vit_model_name="google/vit-base-patch16-224-in21k"
        hidden_dim=4096
        output_dim=256
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.projection_head = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # ADD

        self.classifier= nn.Linear(output_dim, num_classes)



        
    def forward(self, x):
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        projection = self.projection_head(cls_token)
        projection=self.classifier(projection)

        return projection







