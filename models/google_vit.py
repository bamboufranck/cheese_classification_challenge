from transformers import ViTForImageClassification
import torch.nn as nn
import timm

class GoogleVitFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()

        # Charger le modèle pré-entraîné
        self.backbone = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
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
        hidden_dim=768
        dropout_rate=0.5
        """""
        self.classifier = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        """""

        self.classifier= nn.Linear(1024, num_classes)
        

    def forward(self, x):
        outputs = self.backbone(x)  # Exécute le modèle pré-entraîné
        logits = outputs.logits  # Extraire les logits de l'objet de sortie
        x = self.classifier(logits)  # Passer les logits au classificateur
        return x



