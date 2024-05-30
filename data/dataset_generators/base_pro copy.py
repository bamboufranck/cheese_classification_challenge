from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
import random


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



def correct(text,key_word,labels):

    bags=["cheese","cheeses","cake","cakes"]

    for word in bags:
        start = text.find(word)
        if start != -1:
            text = text.replace(word, key_word)

    for word in labels:
        start=text.find(word)
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

        #labels_names_with_cheese = [name + " cheese" for name in labels]
        #text_input = processor(text=labels_names_with_cheese, return_tensors="pt", padding=True)


        m_batch={}
        # ensuite ici fine tune mon générateur avec val_data ou tout simplement utilise ses images 
        # pour mieux generer

        # ou encore utiliser ca pour generer de meilleur prompt avec clip interrogator par exemple 


        labels_prompts,map_images = self.create_prompts(label,val_data,maping,labels)

        image_val_features=processor(images=torch.stack(map_images[label]), return_tensors="pt")
        m_batch[label]=model.get_image_features(**image_val_features)
        m_batch[label]= m_batch[label]/m_batch[label].norm(dim=-1, keepdim=True)


        image_id_0 = 0
        numbers=0
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
                    
                    #text_features = model.get_text_features(**text_input)  # Ajout # Ajout

                    #text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Ajout

                    #similarities = torch.matmul(image_features, text_features.T)  # Ajout #ajout1
                    #predicted_index = similarities.argmax().item()               # Ajout
                    #predicted_category = labels_names_with_cheese[predicted_index] # Ajout




                    ### Ajout avec plutot une similarité avec le val 
                    average_similarity = similarities.mean().item()
                    print("average similarity", average_similarity)

                if(average_similarity>0.50): 
                    numbers+=1
                    print("save!",numbers)                             # Ajout
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
    

    def create_prompts(self, lab,val_data,maping,labels):

        prompts = {}
        map_images={}
        to_pil = transforms.ToPILImage()
        prompts[lab]=[]
        map_images[lab]=[]


        backgrounds = [
            "rustic wooden table", "clean marble countertop", "brick wall", "shelves with wine bottles", 
            "culinary books", "modern kitchen with sleek lines", "neutral-toned cabinets", 
            "large window with curtains", "potted plants", "wall art", "hanging pots and pans", 
            "open shelving", "vintage kitchen tools", "wooden beams", "exposed brickwork", "stone tiles", 
            "farmhouse decor", "chalkboard with menu", "hanging garlic", "string lights", "metal bar stools", 
            "wooden bench", "basket with bread", "hanging herbs", "wine rack", "fruit basket", 
            "cutting boards on wall", "copper pots", "ceramic dishes", "glass jars with grains", 
            "wooden crates", "vintage scales", "mason jars", "hanging dried flowers", "lace tablecloth", 
            "leather bar stools", "bar cart", "metal shelves", "antique kitchen items", 
            "ceramic tile backsplash", "wall clock", "windowsill with plants", "french doors", 
            "glass cabinet doors", "farmhouse sink", "distressed wood furniture", "industrial lighting", 
            "vintage signage", "wooden stools", "iron skillet", "rolling pin", "linen towels", 
            "mixing bowls", "kitchen island", "open pantry", "decorative plates", "fresh produce", 
            "potted succulents", "wooden ladder", "glass domes", "rustic lanterns", "hanging baskets", 
            "vintage jars", "patchwork quilt", "farm tools", "market baskets", "wooden shutters", 
            "exposed pipes", "iron hooks", "wire baskets", "ceramic pitchers", "stone mortar and pestle", 
            "rustic candle holders", "kitchen scales", "old-fashioned oven", "checkered napkins", 
            "wooden trays", "enamel bowls", "herb garden", "metal colander", "handwoven baskets", 
            "ceramic crocks", "hand-painted tiles", "copper kettle", "vintage posters", 
            "retro refrigerator", "cast iron stove", "wooden spoons", "spices in jars", "wrought iron racks", 
            "hanging utensils", "carved wooden signs", "terracotta pots", "antique cutting boards", 
            "galvanized buckets", "tin containers", "slate placemats", "wooden rolling pins", 
            "rustic bread bins", "checkered tablecloths"
        ]

        lighting_conditions = [
            "soft natural light", "bright natural light", "golden hour light", "diffused sunlight", 
            "overcast daylight", "morning sunlight", "afternoon sunlight", "evening sunlight", 
            "candlelight", "warm incandescent light", "cool fluorescent light", "LED lighting", 
            "ambient light", "spotlight", "under-cabinet lighting", "pendant lights", "recessed lighting", 
            "track lighting", "chandelier lighting", "string lights", "table lamp light", "lantern light", 
            "sconce lighting", "floor lamp lighting", "natural window light", "studio lighting", 
            "reflector light", "backlighting", "side lighting", "top-down lighting", "bottom-up lighting", 
            "warm golden light", "soft white light", "bright white light", "cool white light", 
            "soft blue light", "harsh light", "diffused softbox light", "ring light", "flash lighting", 
            "bounce lighting", "fill lighting", "rim lighting", "cross lighting", "natural sunset light", 
            "moonlight", "firelight", "overhead kitchen lights", "skylight illumination", "cafe-style lighting", 
            "bistro lights", "Edison bulb lighting", "soft yellow light", "twilight light", "dawn light", 
            "fireplace light", "halogen lighting", "adjustable lighting", "task lighting", "soft orange light", 
            "gentle shadowing", "minimal shadows", "dramatic shadows", "high contrast light", 
            "low contrast light", "natural daylight", "bright daylight", "subdued lighting", "soft glow", 
            "radiant light", "reflected light", "soft shadows", "hard shadows", "continuous lighting", 
            "daylight temperature light", "tungsten light", "fluorescent temperature light", "cool daylight", 
            "warm afternoon light", "crisp morning light", "gentle evening light", "late afternoon light", 
            "early morning light", "golden morning light", "soft evening glow", "soft focus light", 
            "sharp focus light", "natural ambient light", "diffused window light", "filtered sunlight", 
            "shade lighting", "bright overcast light", "cloudy day light", "interior lighting", 
            "natural kitchen light", "candle-lit ambiance", "string-lit background", "spotlit subject", 
            "low ambient light", "high ambient light"
        ]

        camera_angles = [
            "eye level", "slightly tilted", "close-up", "wide shot", "top-down", "side angle", 
            "45-degree angle", "low angle", "high angle", "overhead", "macro shot", "wide-angle", 
            "panoramic view", "dutch angle", "extreme close-up", "medium shot", "long shot", 
            "close mid shot", "close-up wide shot", "profile shot", "three-quarter view", "from above", 
            "from below", "straight-on", "angled view", "over-the-shoulder", "subject-centered", 
            "off-center", "symmetrical framing", "asymmetrical framing", "low close-up", "high close-up", 
            "wide profile shot", "narrow profile shot", "perspective shot", "diagonal shot", 
            "horizontal framing", "vertical framing", "depth of field focus", "foreground emphasis", 
            "background emphasis", "balanced framing", "leading lines", "rule of thirds", "centered subject", 
            "off-centered subject", "top focus", "bottom focus", "angled from left", "angled from right", 
            "tilted from above", "tilted from below", "parallel framing", "perpendicular framing", 
            "symmetrical balance", "asymmetrical balance", "high perspective", "low perspective", 
            "bird’s eye view", "worm’s eye view", "detail-focused", "broad view", "flat lay", 
            "dynamic angle", "static angle", "compositional focus", "contextual focus", "extreme wide shot", 
            "mid-wide shot", "mid-close shot", "slight upward tilt", "slight downward tilt", 
            "overhead close-up", "side close-up", "off-center close-up", "centered close-up", "pan shot", 
            "tilt shot", "zoom-in", "zoom-out", "rack focus", "pull focus", "push-in", "pull-out", 
            "establishing shot", "action shot", "reaction shot", "insert shot", "point of view", 
            "reverse angle", "multi-angle", "single angle", "static frame", "moving frame", 
            "circular framing", "spiral framing", "square framing", "rectangular framing", 
            "geometric framing", "organic framing"
        ]

        locations = [
    "grocery store", "supermarket", "cheese shop", "farmers' market", "home refrigerator", 
    "cheese counter", "gourmet shop", "restaurant", "restaurant kitchen", 
    "picnic table", "home cheese platter", "charcuterie board", 
    "picnic bag", "wedding buffet", "hotel reception", "office lunch", 
    "holiday dinner", "birthday party", "wine bar", "pastry shop", "market stall", 
    "deli counter", "camping kitchen", "food truck", "dining room", 
    "serving tray", "kitchen counter", "breakfast table", "family reunion", 
    "outdoor terrace", "outdoor meal", "wine tasting", "cellar", "school kitchen", 
    "canteen", "bakery", "catering kitchen", "delivery box", "mountain cabin", 
    "ski resort", "cooking class", "food festival", "tea salon", 
    "forest picnic", "boat kitchen", "airplane (in-flight meal)", "brunch buffet", 
    "stadium concession", "park picnic area", "hotel buffet", "hotel room", 
    "highway rest area", "office refrigerator", "hair salon (small snack)", 
    "mountain lodge", "charity sale", "gala dinner", "outdoor wedding", 
    "neighborhood party", "beer festival", "harvest festival", "beach picnic", 
    "outdoor reception", "Christmas market", "music festival", "trade show", 
    "culinary conference", "theme park", "brewery", "sports club", "street kiosk", 
    "hospital kitchen", "cooking school", "retreat center", "private club", "VIP lounge", 
    "nature reserve", "mountain chalet", "luxury hotel", "spa", "café terrace", 
    "company canteen", "television studio", "tasting event", "agricultural fair", 
    "local market", "family brunch", "gas station", "train (dining car)", "cruise ship", 
    "beach resort", "wedding hall", "holiday meal", "night market", "tapas bar", 
    "conference room", "water park", "hotel breakfast buffet", "classroom meal", 
    "airport lounge"
]


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
        
        """""

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

        """""

        #GPT PROMPT
        cheese_info = {
        "BRIE DE MELUN": {
            "color": ["white", "pale yellow"],
            "shape": ["round"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "CAMEMBERT": {
            "color": ["white", "pale yellow"],
            "shape": ["round"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "EPOISSES": {
            "color": ["orange", "red"],
            "shape": ["round"],
            "crust": ["washed crust"],
            "texture": ["soft", "pungent"]
        },
        "FOURME D’AMBERT": {
            "color": ["ivory", "blue veins"],
            "shape": ["cylindrical"],
            "crust": ["natural crust"],
            "texture": ["creamy", "blue veins"]
        },
        "RACLETTE": {
            "color": ["pale yellow"],
            "shape": ["wheel"],
            "crust": ["natural crust"],
            "texture": ["semi-firm", "melts easily"]
        },
        "MORBIER": {
            "color": ["ivory", "black layer"],
            "shape": ["round"],
            "crust": ["natural crust"],
            "texture": ["semi-soft"]
        },
        "SAINT-NECTAIRE": {
            "color": ["yellow", "orange"],
            "shape": ["round"],
            "crust": ["washed crust"],
            "texture": ["creamy", "smooth"]
        },
        "POULIGNY SAINT- PIERRE": {
            "color": ["white"],
            "shape": ["pyramid"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "ROQUEFORT": {
            "color": ["white", "blue veins"],
            "shape": ["wheel"],
            "crust": ["no crust"],
            "texture": ["crumbly", "creamy"]
        },
        "COMTÉ": {
            "color": ["pale yellow"],
            "shape": ["wheel"],
            "crust": ["natural crust"],
            "texture": ["firm", "smooth"]
        },
        "CHÈVRE": {
            "color": ["white"],
            "shape": ["various"],
            "crust": ["bloomy crust", "natural crust"],
            "texture": ["soft", "firm"]
        },
        "PECORINO": {
            "color": ["pale yellow"],
            "shape": ["wheel"],
            "crust": ["natural crust"],
            "texture": ["hard", "crumbly"]
        },
        "NEUFCHATEL": {
            "color": ["white"],
            "shape": ["heart-shaped"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "CHEDDAR": {
            "color": ["yellow", "orange"],
            "shape": ["block"],
            "crust": ["natural crust", "waxed crust"],
            "texture": ["firm"]
        },
        "BÛCHETTE DE CHÈVRE": {
            "color": ["white"],
            "shape": ["log"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "PARMESAN": {
            "color": ["pale yellow"],
            "shape": ["wheel"],
            "crust": ["natural crust"],
            "texture": ["hard", "granular"]
        },
        "SAINT- FÉLICIEN": {
            "color": ["ivory"],
            "shape": ["round"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "MONT D’OR": {
            "color": ["pale yellow"],
            "shape": ["round"],
            "crust": ["washed crust"],
            "texture": ["soft", "spoonable"]
        },
        "STILTON": {
            "color": ["ivory", "blue veins"],
            "shape": ["cylindrical"],
            "crust": ["natural crust"],
            "texture": ["crumbly"]
        },
        "SCAMORZA": {
            "color": ["white", "pale yellow"],
            "shape": ["pear-shaped"],
            "crust": ["natural crust"],
            "texture": ["semi-soft"]
        },
        "CABECOU": {
            "color": ["white"],
            "shape": ["small round"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "BEAUFORT": {
            "color": ["pale yellow"],
            "shape": ["wheel"],
            "crust": ["natural crust"],
            "texture": ["firm", "smooth"]
        },
        "MUNSTER": {
            "color": ["orange"],
            "shape": ["round"],
            "crust": ["washed crust"],
            "texture": ["soft", "pungent"]
        },
        "CHABICHOU": {
            "color": ["white"],
            "shape": ["small cylindrical"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "TOMME DE VACHE": {
            "color": ["pale yellow"],
            "shape": ["wheel"],
            "crust": ["natural crust"],
            "texture": ["firm"]
        },
        "REBLOCHON": {
            "color": ["pale yellow"],
            "shape": ["small round"],
            "crust": ["washed crust"],
            "texture": ["soft", "creamy"]
        },
        "EMMENTAL": {
            "color": ["pale yellow"],
            "shape": ["large wheel"],
            "crust": ["natural crust"],
            "texture": ["firm", "holes"]
        },
        "FETA": {
            "color": ["white"],
            "shape": ["block"],
            "crust": ["no crust"],
            "texture": ["crumbly"]
        },
        "OSSAU- IRATY": {
            "color": ["pale yellow"],
            "shape": ["wheel"],
            "crust": ["natural crust"],
            "texture": ["firm"]
        },
        "MIMOLETTE": {
            "color": ["orange"],
            "shape": ["ball"],
            "crust": ["natural crust"],
            "texture": ["hard", "crumbly"]
        },
        "MAROILLES": {
            "color": ["orange"],
            "shape": ["square"],
            "crust": ["washed crust"],
            "texture": ["soft", "pungent"]
        },
        "GRUYÈRE": {
            "color": ["pale yellow"],
            "shape": ["wheel"],
            "crust": ["natural crust"],
            "texture": ["firm"]
        },
        "MOTHAIS": {
            "color": ["white"],
            "shape": ["small round"],
            "crust": ["bloomy crust"],
            "texture": ["soft", "creamy"]
        },
        "VACHERIN": {
            "color": ["pale yellow"],
            "shape": ["round"],
            "crust": ["washed crust"],
            "texture": ["soft", "spoonable"]
        },
        "MOZZARELLA": {
            "color": ["white"],
            "shape": ["ball"],
            "crust": ["no crust"],
            "texture": ["soft", "stretchy"]
        },
        "TÊTE DE MOINES": {
            "color": ["pale yellow"],
            "shape": ["small wheel"],
            "crust": ["natural crust"],
            "texture": ["firm"]
        },
        "FROMAGE FRAIS": {
            "color": ["white"],
            "shape": ["various"],
            "crust": ["no crust"],
            "texture": ["soft", "creamy"]
        }
        }

        attributes = cheese_info[lab]
        color = ", ".join(attributes["color"])
        shape = ", ".join(attributes["shape"])
        crust = ", ".join(attributes["crust"])
        texture = ", ".join(attributes["texture"])
        
        base_prompt = (f"Generate an image of {lab} cheese. "
                   f"the {shape} in shape, "
                   f"with a {crust}. It has a {texture} texture.")
        
        for _ in range(50):
            bg = random.choice(backgrounds)
            light = random.choice(lighting_conditions)
            angle = random.choice(camera_angles)
            location=random.choice(locations)
            p= (f"{base_prompt} A photograph of {lab} cheese with a {bg} in the background, "
                  f"illuminated by {light}, captured from a {angle} angle, in {location}.")

            prompts[lab].append({
                    "prompt": p,
                    "num_images": self.num_images_per_label,
                })
        
       
        model_id = "xtuner/llava-phi-3-mini-hf"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        #llava

        prompt1 = "<|user|>\n<image>\nDescribe the image in fifty words, focusing primarily on the cheese; its texture, its shape, its color, its crust and its surroundings.<|end|>\n<|assistant|>\n"
        prompt3= "<|user|>\n<image>\nGive me a  caption of this image.<|end|>\n<|assistant|>\n"
        prompt2 = "<|user|>\n<image>\nDescribe the  cheese in the image,precisely the shape, the color, the crust and the texture.<|end|>\n<|assistant|>\n"

        prompts_liste=[prompt1,prompt3,prompt2]

       


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
                map_images[lab].append(image)
                image = to_pil(image)
                for prompt in prompts_liste:
                    inputs = processor(prompt,image, return_tensors='pt').to(device, torch.float16)
                    output = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.9, top_k=50)
                    description=processor.decode(output[0][2:], skip_special_tokens=True)

                    j=description.find(".")
                    description=description[j+1:]
                    description=correct(description,f" {lab} cheese",labels)
                    description=f"An image of a {lab} cheese," + description
                    print(description)
            
                    prompts[lab].append(
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
