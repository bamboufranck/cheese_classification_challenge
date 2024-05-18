import torchvision.transforms as transforms
import torch
from PIL import Image
from .base import DatasetGenerator
#from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# for blip
#from transformers import AutoProcessor, BlipForConditionalGeneration


#laava 
from transformers import pipeline


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
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """""
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
         
        max_length = 20
        num_beams = 10
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        model.to(device)

        """""

        """""
         
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)

        """""

        model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
        pipe = pipeline("image-to-text", model=model_id, device=device)

        model_id.to(device)
        prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
    

        prompts = {}
        map_images={}
        to_pil = transforms.ToPILImage()

       
        for label in labels_names:
            prompts[label]=[]
            map_images[label]=[]
            prompts[label].append({
                    "prompt": f"an image of {label} cheese",
                    "num_images": self.num_images_per_label,
                })
            
        
        
        
        print( "generation of prompts")

        for i,batch in enumerate(val_data):
            image, label = batch
            valeur_label = label[0].item()
            image = image.squeeze(0)
            map_images[maping[valeur_label]].append(image)
            image = to_pil(image)

            outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

            
            """""
            inputs = blip_processor(images=image, return_tensors="pt").to(device, torch.float16)

            """""

            """""
            pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            """""

            """""
            output_ids = model.generate(pixel_values, **gen_kwargs)
            descriptions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            description= f" A {maping[valeur_label]} cheese" + " and " + descriptions[0].strip()
            """""

            """""

            pixel_values = inputs.pixel_values
            generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=20)
            generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            description=  f" A {maping[valeur_label]} cheese," + generated_caption.split("\n")[0]

            
            """""

            description= f" A {maping[valeur_label]} cheese," + outputs
            prompts[maping[valeur_label]].append(
                {
                    "prompt": description,
                    "num_images": self.num_images_per_label,
                }
            )



            """""
        del blip_model

            """""
        
        del model_id
        torch.cuda.empty_cache()

        print("end of generation")
       
        return prompts,map_images