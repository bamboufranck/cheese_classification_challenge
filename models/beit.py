import torch
import torch.nn as nn
from transformers import BeitForImageClassification

class CustomBeitClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model_name='microsoft/beit-base-patch16-224-pt22k-ft22k', frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.beit = BeitForImageClassification.from_pretrained(pretrained_model_name)

        if frozen:
            for param in self.beit.parameters():
                param.requires_grad = False

            if unfreeze_last_layer:
                # Hypothétiquement, dégeler la dernière couche norm et le dernier block
                # Assurez-vous d'utiliser les bons noms ici basés sur la structure imprimée
                for param in self.beit.beit.encoder.layers[-1].parameters():
                    param.requires_grad = True

        self.beit.classifier = nn.Identity()
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.beit.config.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        outputs = self.beit(x)
        x = outputs.logits
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
