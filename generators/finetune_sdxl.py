import torch
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers import StableDiffusionXLRefinerPipeline
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
            "CAMEMBERT": "Franck19/camembert",
            "EPOISSES": "Franck19/epoisses",
            "FOURME D’AMBERT": "Franck19/fourme",
            "RACLETTE": "Franck19/raclette",
            "MORBIER": "Franck19/morbier",  # If morbier model exists in your setup
            "SAINT-NECTAIRE": "Franck19/saintnect",
            "POULIGNY SAINT- PIERRE": "Franck19/pouligny",  # Assuming naming pattern
            "ROQUEFORT": "Franck19/roquefort",
            "COMTÉ": "Franck19/comte",
            "CHÈVRE": "Franck19/chabichou",  # Assuming as general goat cheese if specific not available
            "PECORINO": "Franck19/pecorino",  # If pecorino model exists in your setup
            "NEUFCHATEL": "Franck19/neufchatel",  # If neufchatel model exists in your setup
            "CHEDDAR": "Franck19/cheddar",
            "BÛCHETTE DE CHÈVRE": "Franck19/buchette",
            "PARMESAN": "Franck19/parmesan",  # If parmesan model exists in your setup
            "SAINT- FÉLICIEN": "Franck19/saintfelicien",  # Assuming naming pattern
            "MONT D’OR": "Franck19/montdor",  # Assuming naming pattern
            "STILTON": "Franck19/stilton",
            "SCARMOZA": "Franck19/scarmoza",
            "CABECOU": "Franck19/cabecou",
            "BEAUFORT": "Franck19/beaufort",
            "MUNSTER": "Franck19/munster",  # If munster model exists in your setup
            "CHABICHOU": "Franck19/chabichou",
            "TOMME DE VACHE": "Franck19/tommedevache",
            "REBLOCHON": "Franck19/reblochon",  # If reblochon model exists in your setup
            "EMMENTAL": "Franck19/emmental",
            "FETA": "Franck19/feta",
            "OSSAU- IRATY": "Franck19/ossau",  # Assuming naming pattern
            "MIMOLETTE": "Franck19/mimolette",  # If mimolette model exists in your setup
            "MAROILLES": "Franck19/maroilles",
            "GRUYÈRE": "Franck19/gruyere",
            "MOTHAIS": "Franck19/mothais",  # If moathais model exists in your setup
            "VACHERIN": "Franck19/vacherin",
            "MOZZARELLA": "Franck19/mozzarella",  # If mozzarella model exists in your setup
            "TÊTE DE MOINES": "Franck19/tetedemoine",
            "FROMAGE FRAIS": "Franck19/fromageFrais"
        }


        self.repo_base = "ByteDance/SDXL-Lightning"
        base = "stabilityai/stable-diffusion-xl-base-1.0"

        self.actual_label=""

        self.pipe = DiffusionPipeline.from_pretrained(
            base, torch_dtype=torch.float16, variant="fp16",
        ).to(device)

        

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.set_progress_bar_config(disable=True)

        
        if use_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        self.num_inference_steps = 50
        self.guidance_scale = 0


        self.pipe.load_lora_weights(self.repo_base)

         # refiner 
       
        refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)

    def generate(self, prompts,label):

        # define a method to update self.repo  using the right label
        if (self.actual_label!=label):
            self.actual_label=label
            self.pipe.load_lora_weights(self.models[label],token=hf_token)

        
        images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images

        refined_output = self.refiner_pipe(prompts, image=images, num_inference_steps=50, guidance_scale=7.5)
        refined_image = refined_output.images[0]

        return refined_image


        

