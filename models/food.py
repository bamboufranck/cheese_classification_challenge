import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import SwinForImageClassification, SwinConfig



class CheeseClassifier(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        model_name = "nateraw/food"  # Correction ici, suppression de la virgule
        #self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        hidden_dim=75
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Unfreeze last layer if required
        if unfreeze_last_layer:
            for param in self.model.classifier.parameters():
                param.requires_grad = True


        
        
        # Update classifier to match the number of classes
        #self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

        self.model.classifier= nn.Sequential(
            nn.Linear(self.model.classifier.in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        # Assuming x is already preprocessed and ready for the model
        #x=self.processor(x)
        #outputs = self.model(pixel_values=x)
        outputs=self.model(x)
        #outputs=self.projection_head(x)
        logits = outputs.logits
        return logits


"""""

class CheeseClassifier(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, dropout_rate=0.5, frozen=True, unfreeze_last_layer=False):
        super().__init__()
        self.config = SwinConfig.from_pretrained('microsoft/swin-base-patch4-window7-224')
        self.config.num_labels = num_classes
        self.backbone = SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224', config=self.config)

        # Freeze all layers if required
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Unfreeze the last layer if required
        if unfreeze_last_layer:
            for param in self.backbone.swin.encoder.layer[-1].parameters():
                param.requires_grad = True

        # Determine the number of features from the backbone
        num_features = self.backbone.classifier.in_features

        # Define the classifier with additional layers
        self.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        outputs = self.backbone(x).pooler_output  # assuming pooler_output is the relevant output
        logits = self.classifier(outputs)
        return logits
     
    """""

