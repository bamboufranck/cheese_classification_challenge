import torch
import torch.nn as nn
from transformers import DeiTModel
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor

class FranckVit(nn.Module):
    def __init__(self, num_classes, frozen=False, unfreeze_last_layer=True):
        super().__init__()
        self.backbone = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.backbone.eval()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if unfreeze_last_layer:
                for param in self.backbone.encoder.layer[-1].parameters():
                    param.requires_grad = True

        self.features_dim = self.backbone.config.hidden_size

        # Pour l'encodage de ce texte avec Sentence-BERT
        self.text_encoder = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        # Classifieur
        self.classifier1 = nn.Linear(self.features_dim + 384, 768)
        self.classifier2 = nn.Linear(768, num_classes)

    def forward(self, x):
        x = x.to(device)
        visual_features = self.backbone(x).last_hidden_state[:, 0]

        # Dénormaliser les images pour Tesseract OCR
        images = denormalize(x)
        features_extractor_text_list = []

        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            generated_text = pytesseract.image_to_string(img_pil)

            # Encoder le texte généré
            features_text = torch.tensor(self.text_encoder.encode(generated_text, convert_to_tensor=True)).to(device)
            features_extractor_text_list.append(features_text)

        features_extractor_text = torch.stack(features_extractor_text_list).to(device)
        combined_features = torch.cat([visual_features, features_extractor_text], dim=1)

        predictions = self.classifier1(combined_features)
        predictions = self.classifier2(predictions)

        return predictions


