from transformers import ViTForImageClassification
import torch.nn as nn
import torchvision.transforms as transforms
import torch
#from transformers import ViTFeatureExtractor
from transformers import ViTImageProcessor, ViTModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BertModel, BertTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"



# Définition des transformations de base sans normalisation

# Charger le modèle et l'extracteur de caractéristiques
#model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
#feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')

def denormalize(tensor):
    # Convertit un tenseur normalisé (ImageNet) en un tenseur avec des valeurs entre 0 et 1
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()

    tensor = tensor * std + mean  # Appliquer l'inverse de la normalisation
    tensor = torch.clamp(tensor, 0.0, 1.0)  # Clamp les valeurs pour s'assurer qu'elles sont entre 0 et 1
    return tensor





class FranckVit(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        # Charger le modèle pré-entraîné pour l'exctraction de features 
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
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
      


        # pour la vision de texte sur l'image
        self.processor_text = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model_text = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

         # pour lencodage de ce texte avec bert
        self.tokenizer= BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')





        # classifieur
        #self.classifier = nn.Linear(1536, num_classes)
        #self.relu=nn.ReLU()
        self.classifier1= nn.Linear(1536, 768) # Ajuster selon les dimensions combinées
        self.classifier2= nn.Linear(768, num_classes)
 

    def forward(self, x):
        x=x.to(device)
        image=denormalize(x)
        x= self.backbone(x)

        features_extractor_text_list = []
        # Traitement image par image pour la génération de texte
    
        for img in image:
            pixel_values = self.processor_text(images=img.unsqueeze(0), return_tensors="pt").pixel_values.to(device)
            generated_ids = self.model_text.generate(pixel_values)
            generated_text = self.processor_text.batch_decode(generated_ids, skip_special_tokens=True,max_length=20)[0]
            
            encoded_input = self.tokenizer(generated_text, return_tensors='pt').to(device)
            output = self.text_encoder(**encoded_input)
            features_text = output.last_hidden_state[:, 0, :]
            features_extractor_text_list.append(features_text.squeeze(0))


        
        
        
        features_extractor_text = torch.stack(features_extractor_text_list).to(device)

        print(image.shape)
        print(features_extractor_text.shape)
            
        combined_features = torch.cat([x, features_extractor_text], dim=1)

       

        predictions = self.classifier1(combined_features)
        predictions = self.classifier2(predictions)

        return predictions

