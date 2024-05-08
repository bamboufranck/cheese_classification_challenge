import torchvision.transforms as transforms
import torch
from PIL import Image
from .base import DatasetGenerator
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer





model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 20
num_beams = 10
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}




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
        prompts = {}
        to_pil = transforms.ToPILImage()
       
        for label in labels_names:
            prompts[label]=[]

        
        
        print( "generation of prompts")

        for i,batch in enumerate(val_data):
            image, label = batch
            image = image.squeeze(0)
            image = to_pil(image)
            valeur_label = label[0].item()
            
          
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

           
            output_ids = model.generate(pixel_values, **gen_kwargs)
            descriptions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            description= f" An image of a piece of {maping[valeur_label]} cheese" + " and " + descriptions[0].strip()

            prompts[maping[valeur_label]].append(
                {
                    "prompt": description,
                    "num_images": self.num_images_per_label,
                }
            )

            torch.cuda.empty_cache()

        print("end of generation")
       
        return prompts