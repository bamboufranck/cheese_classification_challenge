from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms

import torchvision
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration


#for blip
#from transformers import AutoProcessor, BlipForConditionalGeneration, pipeline
import os

from huggingface_hub import login

hf_token= os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("The secret `HF_TOKEN` does not exist in your Colab secrets. Please add it and restart the session.")

# Authentifier avec Hugging Face
login(token=hf_token)



def correct(text,key_word):

    bags=["cheese","cheeses","cake","cakes"]

    for word in bags:
        start = text.find(word)
        if start != -1:
            text = text.replace(word, key_word)

    return text






class DatasetGeneratorFromage:
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
          num_images_per_label=10
    ):
        """
        Args:
            generator: image generator object
            batch_size: Number of images to generate per batch. Make sure to max out your GPU VRAM for maximum efficiency
            output_dir: Directory where the generated images will be saved
        """
        self.generator = generator
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.num_images_per_label = num_images_per_label

    def generate(self, label,val_data,maping):

        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")   # Ajout
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   # Ajout


        m_batch={}
        # ensuite ici fine tune mon générateur avec val_data ou tout simplement utilise ses images 
        # pour mieux generer

        # ou encore utiliser ca pour generer de meilleur prompt avec clip interrogator par exemple 


        labels_prompts,map_images = self.create_prompts(label,val_data,maping)
        image_val_features=processor(images=torch.stack(map_images[label]), return_tensors="pt")
        m_batch[label]=model.get_image_features(**image_val_features)
        m_batch[label]= m_batch[label]/m_batch[label].norm(dim=-1, keepdim=True)


        image_id_0 = 98
        for prompt_metadata in labels_prompts[label]:

            num_images_per_prompt = prompt_metadata["num_images"]
            prompt = [prompt_metadata["prompt"]] * num_images_per_prompt
            pbar = tqdm(range(0, num_images_per_prompt, self.batch_size))
            pbar.set_description(
                    f"Generating images for prompt: {prompt_metadata['prompt']}"
                )
            for i in range(0, num_images_per_prompt, self.batch_size):
                
                batch = prompt[i : i + self.batch_size]
                    
                images = self.generator.generate(batch,label)

                image_input = processor(images=images, return_tensors="pt")  # Ajout

                with torch.no_grad():
                    image_features = model.get_image_features(**image_input) # Ajout
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    similarities = torch.matmul(image_features,  m_batch[label].T)
                        
                        #text_features = model.get_text_features(**text_inputs)  # Ajout # Ajout
                    #text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Ajout

                    #similarities = torch.matmul(image_features, text_features.T)  # Ajout #ajout1
                    #predicted_index = similarities.argmax().item()               # Ajout
                    #predicted_category = labels_names_with_cheese[predicted_index] # Ajout

                    ### Ajout avec plutot une similarité avec le val 
                    average_similarity = similarities.mean().item()
                    print("average similarity", average_similarity)

                    if(average_similarity>=0.55):                              # Ajout
                        self.save_images(images, label, image_id_0)            
                        image_id_0 += len(images)                               
                        pbar.update(1)

                    ### fin 


                    """""
                    if(predicted_category==label+ " cheese"):                              # Ajout
                        self.save_images(images, label, image_id_0)            
                        image_id_0 += len(images)                               
                        pbar.update(1)
                    
                    """""
                        
                    """""
                    self.save_images(images, label, image_id_0)            
                    image_id_0 += len(images)                               
                    pbar.update(1)

                    """""
                    
                    
                pbar.close()
                
       
        del model
        torch.cuda.empty_cache()
    

    def create_prompts(self, lab,val_data,maping):

        #model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model_id = "xtuner/llava-phi-3-mini-hf"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        #blip
        #blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        #blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)

    

    
    
        # llama
        #pipeline = transformers.pipeline("text-generation",model=model_id,tokenizer=model_id,model_kwargs={"torch_dtype": torch.bfloat16})


        #llava

        #prompt = "<|user|>\n<image>\nDescribe the image in sixty words, focusing primarily on the cheese and its surroundings, its location.<|end|>\n<|assistant|>\n"

        prompt = "<|user|>\n<image>\nGenerate a detailed description of the visible elements in this image.<|end|>\n<|assistant|>\n"


        #prompt = "<|user|>\n<image>\n Use this image and generated a detailed prompt, focusing primarily on the cheese and its surroundings.<|end|>\n<|assistant|>\n"
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device, torch.float16)
        processor = AutoProcessor.from_pretrained(model_id)



        prompts = {}
        map_images={}
        to_pil = transforms.ToPILImage()
        prompts[lab]=[]
        map_images[lab]=[]
        
        prompts[lab].append({
                    "prompt": f"an image of {lab} cheese",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f"A rustic wooden platter adorned with a wheel of creamy {lab} cheese , surrounded by fresh grapes, 
figs, and crusty baguette slices",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f"Capture the bustling atmosphere of a French market stall, where {lab} cheese is displayed alongside 
other artisanal cheeses, with handwritten labels and colorful produce",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f"Show the oozy, velvety texture of a warm slice of {lab} cheese, gently melting onto a slice of 
toasted sourdough bread",
                    "num_images": self.num_images_per_label,
                })
        

        prompts[lab].append({
                    "prompt": f"Create an inviting cheeseboard arrangement featuring {lab} cheese, paired with honeycomb, 
walnuts, and a glass of red wine",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f"Place a wheel of {lab} cheese on a vintage marble countertop, with antique silverware and a faded 
French cookbook in the background.",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f" Picture a sun-dappled picnic blanket in a lush garden, where friends share laughter and slices of {lab} cheese with baguette and raspberry jam.",
                    "num_images": self.num_images_per_label,
                })
        


        

        
         
        
        
        

        

            
        
        
        
        print( "start generation of prompts")

       

        for i,batch in tqdm(enumerate(val_data),desc='generation'):
            print("numbers of tours", i)
            image, label = batch
            valeur_label = label[0].item()
            image = image.squeeze(0)
           
            if(maping[valeur_label]==lab):
                map_images[maping[valeur_label]].append(image)
                image = to_pil(image)

                inputs = processor(prompt,image, return_tensors='pt').to(device, torch.float16)
                output = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.9, top_k=50)
                description=processor.decode(output[0][2:], skip_special_tokens=True)

                j=description.find(".")
                description=description[j+1:]
                description=correct(description,f" A {maping[valeur_label]} cheese")
                description=f"An image of a {maping[valeur_label]} cheese," + description
                print(description)
                
                prompts[maping[valeur_label]].append(
                {
                    "prompt": description,
                    "num_images": self.num_images_per_label,
                }
            )
        
        
        return prompts,map_images

            
            # blip
            
        """"
        inputs = blip_processor(images=image, return_tensors="pt").to(device, torch.float16)
        pixel_values = inputs.pixel_values
        generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=60)
        generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        generated_text=generated_caption.split("\n")[0]
        generated_text=correct(generated_text,f" A {maping[valeur_label]} cheese")
        description=f" A {maping[valeur_label]} cheese," + generated_text
        """""
            
        

            # llama
        
            #text="add somes adjectives and some precisions for the following description:" + generated_text 
            #description=  f" A {maping[valeur_label]} cheese," + generated_text


            #description=pipeline(text, max_length=100, num_return_sequences=1,truncation=True)[0]['generated_text']
            

        

            

            # llava 

          


    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
