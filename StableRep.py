import torch
import hydra
from tqdm import tqdm
import torchvision.transforms as transforms
from models.Representation import Encoder
from PIL import Image
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration
import random 
from  generators.finetune_sdxl import FineTune_Sdxl
import torch.nn.functional as F



#configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_images_per_label=4
num_prompt=32
transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(num_ops=10,magnitude=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
epochs=35
tau = 0.5




def correct(text,key_word):

    bags=["cheese","cheeses","cake","cakes"]

    for word in bags:
        start = text.find(word)
        if start != -1:
            text = text.replace(word, key_word)

    return text



def create_prompts(labels_names,val_data,maping):

       
        model_id = "xtuner/llava-phi-3-mini-hf"

        prompt1 = "<|user|>\n<image>\nDescribe the image in fifty words, focusing primarily on the cheese; its texture, its form and its surroundings.<|end|>\n<|assistant|>\n"
        prompt3= "<|user|>\n<image>\nGive me a  caption of this image.<|end|>\n<|assistant|>\n"
        prompt2 = "<|user|>\n<image>\nDescribe the  cheese in the image,precisely the form, the texture and the location also the background of the image.<|end|>\n<|assistant|>\n"

        prompts_liste=[prompt1,prompt3,prompt2]

        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to(device, torch.float16)
        processor = AutoProcessor.from_pretrained(model_id)


        prompts = {}
        map_images={}
        to_pil = transforms.ToPILImage()

        

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
        for label in labels_names:
            prompts[label]=[]
            map_images[label]=[]
            prompts[label].append({
                    "prompt": f"an image of {label} cheese",
                    "num_images": num_images_per_label,
                })


        for lab in labels_names:
            for _ in range(100):
                bg = random.choice(backgrounds)
                light = random.choice(lighting_conditions)
                angle = random.choice(camera_angles)
                location=random.choice(locations)
                prompts[lab].append({
                        "prompt": f"A photograph of {lab} cheese with a {bg} in the background, illuminated by {light}, captured from a {angle} angle, in {location}.",
                        "num_images": num_images_per_label,
                    })
                prompts[lab].append({
            "prompt": f"Illustrate a block of {lab} cheese in a resealable plastic bag, with the packaging showing nutrition facts, a brand logo, and a vibrant product image.",
            "num_images": num_images_per_label,
        })
        
            prompts[lab].append({
                "prompt": f"Generate an image of a round wheel of {lab} cheese wrapped in wax paper, tied with a string, and labeled with a rustic tag. Place it in a cozy kitchen setting.",
                "num_images": num_images_per_label,
            })

            prompts[lab].append({
                "prompt": f"Show a wedge of {lab} cheese in a clear plastic clamshell package, with a price sticker on the front. Display it on a grocery store shelf with other cheese varieties.",
                "num_images": num_images_per_label,
            })

            prompts[lab].append({
                "prompt": f"Depict {lab} cheese crumbles in a transparent tub with a snap-on lid, featuring a colorful brand label and a small scoop inside the container.",
                "num_images": num_images_per_label,
            })

            prompts[lab].append({
                "prompt": f"Create an image of {lab} cheese string sticks individually wrapped, displayed in a branded packaging bag with a vibrant design. Place it on a supermarket shelf.",
                "num_images": num_images_per_label,
            })

            prompts[lab].append({
                "prompt": f"Illustrate a block of {lab} cheese in shrink wrap, with a barcode and a detailed product description. Show it in a kitchen setting with other cooking ingredients.",
                "num_images": num_images_per_label,
            })

            prompts[lab].append({
                "prompt": f"Generate an image of a log of {lab} cheese vacuum-sealed in clear plastic, with a branded label and an expiration date sticker. Place it in a refrigerator section.",
                "num_images": num_images_per_label,
            })

            prompts[lab].append({
            "prompt": f"Show {lab} cheese in a vacuum-sealed package with a clear label that displays the name of the cheese, placed on a supermarket shelf. The packaging should be transparent to show the cheese's texture and color.",
            "num_images": num_images_per_label,
        })

            prompts[lab].append({
                "prompt": f"Depict {lab} cheese in a clear plastic container with a snap-on lid, labeled with the name of the cheese, branding, and nutrition facts, sitting in a refrigerator section with other dairy products.",
                "num_images": num_images_per_label,
            })
            
            prompts[lab].append({
                "prompt": f"Create an image of {lab} cheese slices individually wrapped in clear plastic, each wrapper displaying the name of the cheese, stacked neatly in a packaging box with branding and product information visible.",
                "num_images": num_images_per_label,
            })
            
            prompts[lab].append({
                "prompt": f"Illustrate a block of {lab} cheese in a resealable plastic bag, with the name of the cheese prominently displayed on the packaging along with nutrition facts, a brand logo, and a vibrant product image.",
                "num_images": num_images_per_label,
            })


      
       
        for label in labels_names:
            prompts[label]=[]
            map_images[label]=[]
            prompts[label].append({
                    "prompt": f"an image of {label} cheese",
                    "num_images": num_images_per_label,
                })
            
        
        print( "start generation of prompts")


        for i,batch in tqdm(enumerate(val_data),desc='generation'):
            print("numbers of tours", i)
            image, label = batch
            valeur_label = label[0].item()
            image = image.squeeze(0)
            map_images[maping[valeur_label]].append(image)
            image = to_pil(image)




            for prompt in prompts_liste:
            
                inputs = processor(prompt,image, return_tensors='pt').to(device, torch.float16)
                output = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.9, top_k=50)
                description=processor.decode(output[0][2:], skip_special_tokens=True)
                j=description.find(".")
                description=description[j+1:]
                description=correct(description,f" A {maping[valeur_label]} cheese")
                print(description)

                prompts[maping[valeur_label]].append(
                    {
                        "prompt": description,
                        "num_images": num_images_per_label,
                    }
                )


        del model

        torch.cuda.empty_cache()

        print("end of generation")
       
        return prompts




def compute(prompts,generator,label):

    selected_prompts = random.sample(prompts, num_prompt)

    batch_images = []
    for prompt_metadata in selected_prompts:

        num_images_per_prompt = prompt_metadata["num_images"]
        prompt = [prompt_metadata["prompt"]] * num_images_per_prompt

        images = generator.generate(prompt,label)

        for img in images:
            transformed_image = transformations(img)
            batch_images.append(transformed_image)

    
    batch = torch.stack(batch_images)
    

    return batch

def H(p, q):
    return - (p * torch.log(q)).sum(1).mean()







@hydra.main(config_path="configs/generate", config_name="config")
def generate(cfg):
    valmodule = hydra.utils.instantiate(cfg.get_val)

    val_loaders,maping  = valmodule.val_real_dataloader()

    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

   
    print("stableRep")

    prompts=create_prompts(labels,val_loaders,maping)
    generator=FineTune_Sdxl()
    encoder=Encoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    print("the encoder , the generator and the prompts are build")

    for epoch in tqdm(range(epochs)):

        print("start of the {} number of epochs interation".format(epoch+1))

        for lab, label_prompts in prompts.items():

            batch=compute(label_prompts,generator,lab)
            h=encoder(batch)
            label = torch.arange(num_images_per_label * num_prompt)
            p = (label.view(-1, 1) == label.view(1,-1)).float()
            p.fill_diagonal(0) # self masking
            p /= p.sum(1, keepdim=True)
            # compute contrastive distribution q
            logits = h @ h.T / tau
            logits.fill_diagonal(-1e9) # self masking
            q = F.softmax(logits, dim=1)
            loss = H(p, q)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("end of the {} number of epochs interation".format(epoch+1))



        
        print("end")
        model_save_path="stable_rep_encoder.pth"
        torch.save(encoder.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")
       

        

         
            







    












if __name__ == "__main__":
    generate()
