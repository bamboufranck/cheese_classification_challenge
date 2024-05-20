import torchvision.transforms as transforms
import torch
from PIL import Image
from .base import DatasetGenerator
#from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# for blip
from transformers import AutoProcessor, BlipForConditionalGeneration, pipeline
import os

from huggingface_hub import login
import transformers

hf_token= os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("The secret `HF_TOKEN` does not exist in your Colab secrets. Please add it and restart the session.")

# Authentifier avec Hugging Face
login(token=hf_token)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"



#hf_token = "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"

# Ajouter le token dans les variables d'environnement
#os.environ["HUGGINGFACE_TOKEN"] = hf_token




def correct(text,key_word):

    bags=["cake","cheese","cakes","cheeses"]

    for word in bags:
        start = text.find(word)
        if start != -1:
            text = text.replace(word, key_word)

    return text





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

    
        blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)
        pipeline = transformers.pipeline("text-generation",model=model_id,tokenizer=model_id,model_kwargs={"torch_dtype": torch.bfloat16},use_auth_token=hf_token)
        

        """""
        
        model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
        prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        processor = AutoProcessor.from_pretrained(model_id)
        """""
       


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
            
            # blip

            inputs = blip_processor(images=image, return_tensors="pt").to(device, torch.float16)
            pixel_values = inputs.pixel_values
            generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=60)
            generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            generated_text=generated_caption.split("\n")[0]
            generated_text=correct(generated_text,f" A {maping[valeur_label]} cheese")

            text="add somes adjectives and some precisions for the following description:" + generated_text 
            #description=  f" A {maping[valeur_label]} cheese," + generated_text


            description=pipeline(text, max_length=100, num_return_sequences=1)[0]['generated_text']

            

            # llava 
            """""
            inputs = processor(prompt, image, return_tensors='pt')
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            description=processor.decode(output[0][2:], skip_special_tokens=True)
            """""



            
            prompts[maping[valeur_label]].append(
                {
                    "prompt": description,
                    "num_images": self.num_images_per_label,
                }
            )



    
        del blip_model

        del pipeline

        torch.cuda.empty_cache()

        print("end of generation")
       
        return prompts,map_images