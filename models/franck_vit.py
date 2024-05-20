from transformers import ViTForImageClassification
import torch.nn as nn
import torch
#from transformers import ViTFeatureExtractor
from transformers import ViTImageProcessor, ViTModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BertModel, BertTokenizer

# Charger le modèle et l'extracteur de caractéristiques
#model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
#feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')

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
        #feature extractor for image
        inputs = self.processor_image(images=x, return_tensors="pt")
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

