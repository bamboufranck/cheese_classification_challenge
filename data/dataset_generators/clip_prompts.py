import torchvision.transforms as transforms
import torch
from PIL import Image
from .base import DatasetGenerator
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


class ClipPromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=10,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label


    def create_prompts(self, labels_names,val_data,maping):
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
         
       
        gen_kwargs = {
            "max_length": 20,
            "num_beams": 10,
            "temperature": 0.9,  # Augmenter pour plus de diversité
            "top_k": 50,  # Réduire le nombre pour plus de diversité
            "top_p": 0.92,  # Augmenter pour plus de diversité
            "no_repeat_ngram_size": 2  # Pour éviter la répétition de n-grams
        }
         
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        prompts = {}
        to_pil = transforms.ToPILImage()

       
        for label in labels_names:
            prompts[label]=[]
            prompts[label].append({
                    "prompt": f"an image of {label} cheese",
                    "num_images": self.num_images_per_label,
                })

        
        
        print( "generation of prompts")

        for i,batch in enumerate(val_data):
            image, label = batch
            image = image.squeeze(0)
            image = to_pil(image)
            valeur_label = label[0].item()
            
          
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            attention_mask = pixel_values.new_ones(pixel_values.shape[:2], dtype=torch.long) 

            output_ids = model.generate(pixel_values, attention_mask=attention_mask, **gen_kwargs)
           
            descriptions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            description= f" An image of a piece of {maping[valeur_label]} cheese" + " and " + descriptions[0].strip()

            prompts[maping[valeur_label]].append(
                {
                    "prompt": description,
                    "num_images": self.num_images_per_label,
                }
            )

        del model
        torch.cuda.empty_cache()

        print("end of generation")
       
        return prompts