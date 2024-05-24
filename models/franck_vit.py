import torch
import torch.nn as nn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import BertModel, BertTokenizer
from transformers import DeiTModel

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # Charger le modèle pré-entraîné pour l'extraction de features
        self.backbone = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.backbone.eval()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.backbone.encoder.layer[-1].parameters():
                    param.requires_grad = True

        self.features_dim = self.backbone.config.hidden_size

        # pour la vision de texte sur l'image
        self.processor_text = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', do_rescale=False)
        self.model_text = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')

        # pour l'encodage de ce texte avec BERT
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')

        # classifieur
        self.classifier1 = nn.Linear(self.features_dim + 768, 768) # Ajuster selon les dimensions combinées
        self.classifier2 = nn.Linear(768, num_classes)

    def forward(self, x):
        x = x.to(device)

        # Extraction des caractéristiques visuelles
        visual_features = self.backbone(x).last_hidden_state[:, 0]  # Utiliser les caractéristiques du token [CLS]

        # Dénormaliser les images pour le modèle TrOCR
        images = denormalize(x)
        features_extractor_text_list = []

        # Traitement des images pour générer du texte
        for img in images:
            pixel_values = self.processor_text(images=img.unsqueeze(0), return_tensors="pt").pixel_values.to(device)
            generated_ids = self.model_text.generate(pixel_values, max_new_tokens=25)
            generated_text = self.processor_text.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Encoder le texte généré
            encoded_input = self.tokenizer(generated_text, return_tensors='pt').to(device)
            output = self.text_encoder(**encoded_input)
            features_text = output.last_hidden_state[:, 0, :]
            features_extractor_text_list.append(features_text.squeeze(0))

        # Combiner les caractéristiques visuelles et textuelles
        features_extractor_text = torch.stack(features_extractor_text_list).to(device)
        combined_features = torch.cat([visual_features, features_extractor_text], dim=1)

        # Passer les caractéristiques combinées à travers le classifieur
        predictions = self.classifier1(combined_features)
        predictions = self.classifier2(predictions)

        return predictions

