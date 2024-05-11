from transformers import  ViTForImageClassification
import torch.nn as nn

class GoogleVitFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        # Charger le modèle pré-entraîné de Hugging Face
        self.backbone = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        # Remplacer la tête de classification par une identité pour garder les caractéristiques du modèle
        self.backbone.classifier = nn.Identity()
        
        if frozen:
            # Geler tous les paramètres du modèle
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Optionnellement dégeler la dernière couche de normalisation et le dernier bloc transformer
            if unfreeze_last_layer:
                # Dégeler les paramètres de la dernière couche de normalisation
                for param in self.backbone.vit.layernorm.parameters():
                    param.requires_grad = True
                
                # Dégeler les paramètres du dernier bloc Transformer
                for param in self.backbone.vit.encoder.layer[-1].parameters():
                    param.requires_grad = True

        # Ajouter un classificateur personnalisé à la sortie du modèle
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, x):
        # Passer l'input à travers le backbone
        outputs = self.backbone(x)
        x = outputs.logits  # Utiliser les logits comme nouvelles features
        
        # Passer les features à travers le classificateur
        x = self.classifier(x)
        return x
