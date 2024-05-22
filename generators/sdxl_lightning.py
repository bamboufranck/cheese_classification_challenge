import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from diffusers import StableDiffusionXLImg2ImgPipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"



class SDXLLightiningGenerator:
    def __init__(
        self,
        use_cpu_offload=False,
    ):
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
            device, torch.float16
        )
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.set_progress_bar_config(disable=True)
        if use_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        self.num_inference_steps = 40
        self.guidance_scale = 0
        
        # refiner 
        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)
        




    def generate(self, prompts,label):
        
        images = self.pipe(
            prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images

        refined_output = self.refiner_pipe(prompts, image=images, num_inference_steps=50, guidance_scale=7.5).images

        return refined_output
