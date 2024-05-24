import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

class CheeseClassifier(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        model_name = "nateraw/food"  # Correction ici, suppression de la virgule
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last layer if required
        if unfreeze_last_layer:
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        
        # Update classifier to match the number of classes
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
    
    def forward(self, x):
        # Assuming x is already preprocessed and ready for the model
        x=self.processor(x)
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        return logits
