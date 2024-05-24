import torchvision
import torch.nn as nn
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForImageClassification

class CheeseClassifier(nn.Module):
    def __init__(self,num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        model_name ="nateraw/food",
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        
        # Dummy backbone to match the example structure
        # Replace this with actual model implementation if needed
        self.backbone = nn.Identity()  
        self.features_dim = 101
        self.classifier = nn.Linear(self.features_dim,num_classes)
    
    def forward(self, x):
        x = self.processor(x)
        x = self.model(x)
        x=self.classifier(x)
        return x