import torch
import torch.nn as nn
from google_vit import GoogleVitFinetune
from dinov2 import DinoV2Finetune


class EnsembleModel(nn.Module):
    def __init__(self,num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        model1=GoogleVitFinetune(num_classes, frozen=False, unfreeze_last_layer=True)
        model2=DinoV2Finetune(num_classes, frozen=False, unfreeze_last_layer=True)
        self.models = nn.ModuleList([model1,model2])

    def forward(self, x):
        predictions = []
        for model in self.models:
            logits = model(x)
            predictions.append(logits)
        ensemble_logits = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_logits
