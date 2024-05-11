
from .base_generator import BaseGenerator

class SDXLLightiningGenerator(BaseGenerator):
    def __init__(self, use_cpu_offload=False):
        super().__init__(
            base="stabilityai/stable-diffusion-xl-base-1.0",
            unet_config_path="unet",
            ckpt_path="sdxl_lightning_4step_unet.safetensors",
            use_cpu_offload=use_cpu_offload,
            num_inference_steps=4,
            guidance_scale=0
        )
