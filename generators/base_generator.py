import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

class BaseGenerator:
    def __init__(self, base, unet_config_path, ckpt_path, use_cpu_offload=False, num_inference_steps=4, guidance_scale=7.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(self.device, torch.float16)
        
        if ckpt_path:
            self.unet.load_state_dict(load_file(hf_hub_download(base, ckpt_path), device=self.device))
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base,
            unet=self.unet,
            revision="fp16",
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing"
        )
        
        self.pipe.set_progress_bar_config(disable=True)
        
        if use_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def generate(self, prompts):
        return self.pipe(
            prompt=prompts,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
