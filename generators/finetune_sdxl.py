import torch
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
)
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import os 
from diffusers import StableDiffusionXLImg2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

from huggingface_hub import login

hf_token= os.getenv("HF_TOKEN")

if hf_token is None:
    raise ValueError("The secret `HF_TOKEN` does not exist in your Colab secrets. Please add it and restart the session.")

# Authentifier avec Hugging Face
login(token=hf_token)






class FineTune_Sdxl:
    def __init__(
        self,
        use_cpu_offload=False,
    ):
        self.models = {
            "BRIE DE MELUN": "Franck19/brie",
            "SAINT-FÉLICIEN": "Franck19/saintfeli",
            "CAMEMBERT": "Franck19/camembert",
            "EPOISSES": "Franck19/epoisses",
            "FOURME D’AMBERT": "Franck19/fourme",
            "RACLETTE": "Franck19/raclette",
            "SAINT-NECTAIRE": "Franck19/saintnect",
            "ROQUEFORT": "Franck19/roquefort",
            "COMTÉ": "Franck19/comte",
            "CHÈVRE": "Franck19/chevre",  # Assuming as general goat cheese if specific not available
            "CHEDDAR": "Franck19/cheddar",
            "BÛCHETTE DE CHÈVRE": "Franck19/buchette",
            "STILTON": "Franck19/stilton",
            "SCARMOZA": "Franck19/scarmoza",
            "CABECOU": "Franck19/cabecou",
            "BEAUFORT": "Franck19/beaufort",
            "CHABICHOU": "Franck19/chabichou",
            "TOMME DE VACHE": "Franck19/tommedevache",
            "EMMENTAL": "Franck19/emmental",
            "FETA": "Franck19/feta",
            "MIMOLETTE": "Franck19/mimolette",  # If mimolette model exists in your setup
            "MAROILLES": "Franck19/maroilles",
            "GRUYÈRE": "Franck19/gruyere",
            "VACHERIN": "Franck19/vacherin",
            "TÊTE DE MOINES": "Franck19/tetedemoine",
            "FROMAGE FRAIS": "Franck19/fromageFrais",
            "REBLOCHON": "Franck19/reblochon",
            "PARMESAN": "Franck19/parmesan",
            "POULIGNY SAINT- PIERRE": "Franck19/pouligny",
            "PECORINO": "Franck19/pecorino",
            "NEUFCHATEL": "Franck19/neufchatel",
            "MUNSTER": "Franck19/munster",
            "OSSAU- IRATY":"Franck19/ossau",
            "MOTHAIS": "Franck19/mothais",
            "MORBIER": "Franck19/morbier",
            "MOZZARELLA" :"Franck19/mozarella",
            "MONT D’OR" :"Franck19/montdor"

        }
        base = "stabilityai/stable-diffusion-xl-base-1.0"

        self.actual_label=""

        self.pipe = DiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16",).to(device,torch.float16)

        #self.pipe=DiffusionPipeline.from_pretrained("SG16222/Realistic_Vision_V1.4", torch_dtype=torch.float16, variant="fp16",).to(device,torch.float16)

        

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.set_progress_bar_config(disable=True)

        
        if use_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        self.num_inference_steps = 4
        self.guidance_scale = 7.5

         # refiner 
       
        refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        #self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)

    def generate(self, prompts,label):

        # define a method to update self.repo  using the right label
        if label in self.models:
            if label!=self.actual_label:
                self.actual_label=label
                self.pipe.load_lora_weights(self.models[label],token=hf_token)
                print("load",label)
        
            for index,text in enumerate(prompts):
                text=text.replace(label,"tok")
                prompts[index]=text
                print(prompts[index])
           

            print("start of generation")
            images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
            
            #for index,text in enumerate(prompts):
               # text=text.replace("tok",label)
               # prompts[index]=text
               # print(prompts[index])

        else:
            print("start of generation")
            images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
            
        #print("rafinage")
        
        #refined_output = self.refiner_pipe(prompts, image=images, num_inference_steps=4, guidance_scale=0)
        #refined_image = refined_output.images
       

        print("end generation")

        #return refined_image
        return images


        

