import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


device = "cuda" if torch.cuda.is_available() else "cpu"


class SDXLTurboGenerator:
    def __init__(self, model_name="stabilityai/sdxl-turbo", use_cpu_offload=False):
        # Charger le modèle UNet personnalisé pour SDXL Turbo
        self.unet = UNet2DConditionModel.from_pretrained(model_name).to(device)
        
        # Charger le pipeline Stable Diffusion avec le UNet personnalisé
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            unet=self.unet,
            revision="fp16",  # Assurez-vous que 'revision' correspond aux besoins
            torch_dtype=torch.float16
        ).to(device)

        if use_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        # Configuration par défaut
        self.num_inference_steps = 4
        self.guidance_scale = 7.5
        self.pipe.scheduler.set_timesteps(self.num_inference_steps)

    def generate(self, prompts):
        return self.pipe(
            prompt=prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images