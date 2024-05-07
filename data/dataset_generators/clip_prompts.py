import torch
from .base import DatasetGenerator
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torchvision.transforms as transforms


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")






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

       
        for i,batch in enumerate(val_data):
            image, label = batch
            image = image.squeeze(0)
            image = to_pil(image)
            valeur_label = label[0].item()
            text= f"An image of a {maping[valeur_label]} cheese."

            print("process time")
            prompt = " a description of this image ..."
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
           
            print("generation time")
            input_text = prompt + " " + tokenizer.decode(pixel_values[0], skip_special_tokens=True)+ ". " + text
            generated_ids = model.generate(input_text,max_length=50)
            generated_prompt = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print("end of description")

            prompts[maping[valeur_label]].append(
                {
                    "prompt": generated_prompt,
                    "num_images": self.num_images_per_label,
                }
            )
            torch.cuda.empty_cache()
        return prompts