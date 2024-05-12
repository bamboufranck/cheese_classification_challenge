from transformers import ViTForImageClassification
import torch.nn as nn

class GoogleVitFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        # Charger le modèle pré-entraîné
        self.backbone = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        # Remplacer la tête de classification par une identité (pas de calcul)
        self.backbone.classifier = nn.Identity()

        if frozen:
            # Geler tous les paramètres du modèle
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            if unfreeze_last_layer:
                # Dégelez la dernière couche de normalisation, s'il y en a une spécifique
                if hasattr(self.backbone, 'layernorm'):
                    for param in self.backbone.layernorm.parameters():
                        param.requires_grad = True
                
                # Dégelez les paramètres du dernier bloc transformer
                if hasattr(self.backbone, 'vit'):
                    for param in self.backbone.vit.encoder.layer[-1].parameters():
                        param.requires_grad = True

        # Ajouter un nouveau classificateur
        self.classifier = nn.Linear(768, num_classes)  # 768 est la dimension typique pour 'vit-base'

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
