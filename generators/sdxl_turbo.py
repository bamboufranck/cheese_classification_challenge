
from .base_generator import BaseGenerator

class SDXLTurboGenerator(BaseGenerator):
    def __init__(self, use_cpu_offload=False):
        super().__init__(
            base="stabilityai/sdxl-turbo",
            unet_config_path="unet",
            ckpt_path="stabilityai/sdxl-turbo/sdxl_turbo_checkpoint.safetensors",
            use_cpu_offload=use_cpu_offload,
            num_inference_steps=4,
            guidance_scale=7.5
        )