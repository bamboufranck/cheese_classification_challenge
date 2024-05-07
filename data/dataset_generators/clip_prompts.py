import torch
from .base import DatasetGenerator
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torchvision.transforms as transforms

# Charger le modèle et le processeur BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")




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


    def create_prompts(self, labels_names,val_data):
        prompts = {}
        to_pil = transforms.ToPILImage()
       
        for label in labels_names:
            print(label)
            prompts[label]=[]

        for i,batch in enumerate(val_data):
            image, label = batch
            print(label +str(1))
            image = image.squeeze(0)
            image = to_pil(image)
            inputs = processor(images=image, return_tensors="pt")
            output_ids = model.generate(**inputs)
            description = processor.decode(output_ids[0], skip_special_tokens=True)
            prompts[label].append(
                {
                    "prompt": description,
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts