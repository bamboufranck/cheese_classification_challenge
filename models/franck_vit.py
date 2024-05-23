from transformers import ViTForImageClassification
import torch.nn as nn
import torchvision.transforms as transforms
import torch
#from transformers import ViTFeatureExtractor
from transformers import ViTImageProcessor, ViTModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BertModel, BertTokenizer



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
        self.processor_image = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model_image= ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')


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
        #to_pil = transforms.ToPILImage()
        #feature extractor for image
        #x=x.squeeze(0)
        #x=to_pil(x)

        image=denormalize(x)
        inputs = self.processor_image(images=image, return_tensors="pt")
        outputs = self.model_image(**inputs)
        features_extractor_image = outputs.last_hidden_state[:, 0]



        # troc for extracting text in the image
        pixel_values = self.processor_text(images=x, return_tensors="pt").pixel_values
        generated_ids = self.model_text.generate(pixel_values)
        generated_text =self.processor_text.batch_decode(generated_ids, skip_special_tokens=True)[0]
 
        # we encode the text in the image
        encoded_input = self.tokenizer(generated_text, return_tensors='pt')
        output = self.text_encoder(**encoded_input)
        features_extractor_text= output.last_hidden_state[:, 0]

        combined_features = torch.cat([features_extractor_image, features_extractor_text], dim=1)
        #predictions = self.classifier(combined_features)

        predictions = self.classifier1(combined_features)
        predictions = self.classifier2(predictions)


        return predictions

