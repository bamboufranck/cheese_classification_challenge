import torch
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers import StableDiffusionXLRefinerPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

class FineTune_Sdxl:
    def __init__(
        self,
        use_cpu_offload=False,
    ):
        names=["Franck19/chabichou","Franck19/camembert","Franck19/cabecou","Franck19/brie","Franck19/buchette","Franck19/beaufort"]
        names_repos={}
        self.pipes={}
        repo_base = "ByteDance/SDXL-Lightning"
        base = "stabilityai/stable-diffusion-xl-base-1.0"

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



        self.pipe.load_lora_weights(repo_base)




         # refiner 
        refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        self.refiner_pipe = StableDiffusionXLRefinerPipeline.from_pretrained(refiner_model_id, torch_dtype=torch.float16).to(device)


    def generate(self, prompts,label):

        # define a method to update self.repo  using the right label 
        self.pipe.load_lora_weights(self.repo)
        images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images

        refined_output = self.refiner_pipe(prompts, image=images, num_inference_steps=50, guidance_scale=7.5)
        refined_image = refined_output.images[0]

        return refined_image


        

