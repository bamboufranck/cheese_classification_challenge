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

    def generate(self, label,labels,val_data,maping):

        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")   # Ajout
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   # Ajout

        labels_names_with_cheese = [name + " cheese" for name in labels]
        text_input = processor(text=labels_names_with_cheese, return_tensors="pt", padding=True)


        m_batch={}
        # ensuite ici fine tune mon générateur avec val_data ou tout simplement utilise ses images 
        # pour mieux generer

        # ou encore utiliser ca pour generer de meilleur prompt avec clip interrogator par exemple 


        labels_prompts,map_images = self.create_prompts(label,val_data,maping)

        #image_val_features=processor(images=torch.stack(map_images[label]), return_tensors="pt")
        #m_batch[label]=model.get_image_features(**image_val_features)
        #m_batch[label]= m_batch[label]/m_batch[label].norm(dim=-1, keepdim=True)


        image_id_0 = 0
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
                    #similarities = torch.matmul(image_features,  m_batch[label].T)
                    
                    text_features = model.get_text_features(**text_input)  # Ajout # Ajout

                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Ajout

                    similarities = torch.matmul(image_features, text_features.T)  # Ajout #ajout1
                    predicted_index = similarities.argmax().item()               # Ajout
                    predicted_category = labels_names_with_cheese[predicted_index] # Ajout




                    ### Ajout avec plutot une similarité avec le val 
                    #average_similarity = similarities.mean().item()
                    #print("average similarity", average_similarity)

                    if(predicted_category==label+ " cheese"):                              # Ajout
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

        prompts = {}
        map_images={}
        to_pil = transforms.ToPILImage()
        prompts[lab]=[]
        map_images[lab]=[]

        prompts[lab].append({
        "prompt": f"Generate an image of a round wheel of {lab} cheese, showing its texture and color. Place it on a wooden board with a cheese knife beside it.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Create an image of a wedge of {lab} cheese, showcasing its unique characteristics such as holes, veins, or creamy interior. Arrange it on a marble slab with some complementary food items like fruits, nuts, or crackers.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Illustrate a block of {lab} cheese, with a few slices cut off to reveal its interior. Set it on a rustic cutting board with a bunch of grapes and a small jar of honey.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Produce a picture of a soft, spreadable {lab} cheese, displayed in a small bowl or on a piece of crusty bread. Garnish with fresh herbs and place a few olives or cherry tomatoes around it.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Design an image of a whole wheel of {lab} cheese, partially sliced to show the texture inside. Surround it with some fresh fruit slices, a handful of nuts, and a sprig of rosemary.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Create a scene featuring a log of {lab} cheese, garnished with spices or herbs. Place it on a ceramic plate with some sliced baguette and a drizzle of olive oil.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Generate an image of a semi-hard {lab} cheese wheel, with a small wedge cut out to reveal its smooth interior. Set it on a slate serving platter with some fresh berries and a small bowl of jam.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Produce an image of a chunk of {lab} cheese, with a grater nearby and a pile of freshly grated cheese. Include a sprig of fresh basil or thyme in the background.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Illustrate a crumbled {lab} cheese on a simple white plate, with a few olives and cherry tomatoes around it. Drizzle with olive oil and sprinkle with a pinch of herbs.",
        "num_images": self.num_images_per_label,
    })
        prompts[lab].append({
        "prompt": f"Create an image of a fresh {lab} cheese ball, showing its smooth and shiny exterior. Place it on a cutting board with fresh basil leaves, sliced tomatoes, and a small bowl of balsamic vinegar.",
        "num_images": self.num_images_per_label,
    })

        prompts[lab].append({
                    "prompt": f"an image of {lab} cheese",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f"A rustic wooden platter adorned with a wheel of creamy {lab} cheese , surrounded by fresh grapes, figs, and crusty baguette slices",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f"Capture the bustling atmosphere of a French market stall, where {lab} cheese is displayed alongside other artisanal cheeses, with handwritten labels and colorful produce",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f"Show the oozy, velvety texture of a warm slice of {lab} cheese, gently melting onto a slice of toasted sourdough bread",
                    "num_images": self.num_images_per_label,
                })
        

        prompts[lab].append({
                    "prompt": f"Create an inviting cheeseboard arrangement featuring {lab} cheese, paired with honeycomb, walnuts, and a glass of red wine",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f"Place a wheel of {lab} cheese on a vintage marble countertop, with antique silverware and a faded French cookbook in the background.",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
                    "prompt": f" Picture a sun-dappled picnic blanket in a lush garden, where friends share laughter and slices of {lab} cheese with baguette and raspberry jam.",
                    "num_images": self.num_images_per_label,
                })
        
        prompts[lab].append({
        "prompt": f"Show a platter with assorted {lab} cheese slices arranged in a fan shape, accompanied by dried apricots, almonds, and a small dish of honey.",
        "num_images": self.num_images_per_label,
    })

        prompts[lab].append({
            "prompt": f"Depict a picnic scene with a block of {lab} cheese partially sliced, set on a checkered cloth with a loaf of bread, a bottle of wine, and some wildflowers.",
            "num_images": self.num_images_per_label,
        })
        
        prompts[lab].append({
            "prompt": f"Create an image of {lab} cheese cubes skewered with toothpicks, displayed on a wooden platter with assorted berries and a sprig of mint.",
            "num_images": self.num_images_per_label,
        })
        
        prompts[lab].append({
            "prompt": f"Illustrate a close-up shot of a melting {lab} cheese fondue in a pot, with bread cubes on skewers ready to dip, surrounded by assorted vegetables.",
            "num_images": self.num_images_per_label,
        })
        
        prompts[lab].append({
            "prompt": f"Generate an image of a sophisticated cheese board featuring {lab} cheese, along with assorted charcuterie, olives, nuts, and a small bowl of mustard.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Show a festive arrangement of {lab} cheese cut into star shapes, placed on a platter with decorative holiday elements like pine cones and cranberries.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Depict a rustic kitchen setting with a large wheel of {lab} cheese on a wooden table, surrounded by fresh herbs, a cutting board, and kitchen utensils.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Create an image of {lab} cheese being grated over a pasta dish, with a rich and creamy sauce visible in the background.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Illustrate a gourmet setting with a slice of {lab} cheese on a piece of slate, drizzled with truffle oil and garnished with edible flowers.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Generate an image of a cheese market stall showcasing a variety of {lab} cheese, with price tags, a scale, and fresh produce in the background.",
            "num_images": self.num_images_per_label,
        })


        prompts[lab].append({
        "prompt": f"Show {lab} cheese in a vacuum-sealed package with a clear label, placed on a supermarket shelf. The packaging should be transparent to show the cheese's texture and color.",
        "num_images": self.num_images_per_label,
    })

        prompts[lab].append({
            "prompt": f"Depict {lab} cheese in a clear plastic container with a snap-on lid, labeled with branding and nutrition facts, sitting in a refrigerator section with other dairy products.",
            "num_images": self.num_images_per_label,
        })
        
        prompts[lab].append({
            "prompt": f"Create an image of {lab} cheese slices individually wrapped in clear plastic, stacked neatly in a packaging box with branding and product information visible.",
            "num_images": self.num_images_per_label,
        })
        
        prompts[lab].append({
            "prompt": f"Illustrate a block of {lab} cheese in a resealable plastic bag, with the packaging showing nutrition facts, a brand logo, and a vibrant product image.",
            "num_images": self.num_images_per_label,
        })
        
        prompts[lab].append({
            "prompt": f"Generate an image of a round wheel of {lab} cheese wrapped in wax paper, tied with a string, and labeled with a rustic tag. Place it in a cozy kitchen setting.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Show a wedge of {lab} cheese in a clear plastic clamshell package, with a price sticker on the front. Display it on a grocery store shelf with other cheese varieties.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Depict {lab} cheese crumbles in a transparent tub with a snap-on lid, featuring a colorful brand label and a small scoop inside the container.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Create an image of {lab} cheese string sticks individually wrapped, displayed in a branded packaging bag with a vibrant design. Place it on a supermarket shelf.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Illustrate a block of {lab} cheese in shrink wrap, with a barcode and a detailed product description. Show it in a kitchen setting with other cooking ingredients.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
            "prompt": f"Generate an image of a log of {lab} cheese vacuum-sealed in clear plastic, with a branded label and an expiration date sticker. Place it in a refrigerator section.",
            "num_images": self.num_images_per_label,
        })

        prompts[lab].append({
        "prompt": f"Show {lab} cheese in a vacuum-sealed package with a clear label that displays the name of the cheese, placed on a supermarket shelf. The packaging should be transparent to show the cheese's texture and color.",
        "num_images": self.num_images_per_label,
    })

        prompts[lab].append({
            "prompt": f"Depict {lab} cheese in a clear plastic container with a snap-on lid, labeled with the name of the cheese, branding, and nutrition facts, sitting in a refrigerator section with other dairy products.",
            "num_images": self.num_images_per_label,
        })
        
        prompts[lab].append({
            "prompt": f"Create an image of {lab} cheese slices individually wrapped in clear plastic, each wrapper displaying the name of the cheese, stacked neatly in a packaging box with branding and product information visible.",
            "num_images": self.num_images_per_label,
        })
        
        prompts[lab].append({
            "prompt": f"Illustrate a block of {lab} cheese in a resealable plastic bag, with the name of the cheese prominently displayed on the packaging along with nutrition facts, a brand logo, and a vibrant product image.",
            "num_images": self.num_images_per_label,
        })

       
        model_id = "xtuner/llava-phi-3-mini-hf"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        #llava

        prompt1 = "<|user|>\n<image>\nDescribe the image in fifty words, focusing primarily on the cheese; its texture, its form and its surroundings.<|end|>\n<|assistant|>\n"
        prompt2= "<|user|>\n<image>\nInspire you of this image and generate me a prompt in fifty words, which describe primarily the cheese; its texture, its form and its surroundings.<|end|>\n<|assistant|>\n"

        prompts=[prompt1,prompt2]

        #prompt = "<|user|>\n<image>\nDescribe the  cheese in the image,precisely the form, the texture and the location also the background of the image.<|end|>\n<|assistant|>\n"


        #prompt = "<|user|>\n<image>\n Use this image and generated a detailed prompt, focusing primarily on the cheese and its surroundings.<|end|>\n<|assistant|>\n"
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device, torch.float16)
        processor = AutoProcessor.from_pretrained(model_id)
   

        print( "start generation of prompts")

       

        for i,batch in tqdm(enumerate(val_data),desc='generation'):
            print("numbers of tours", i)
            image, label = batch
            valeur_label = label[0].item()
            image = image.squeeze(0)
           
            if(maping[valeur_label]==lab):
                map_images[maping[valeur_label]].append(image)
                image = to_pil(image)
                #for prompt in prompts:
                inputs = processor(prompt1,image, return_tensors='pt').to(device, torch.float16)
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
