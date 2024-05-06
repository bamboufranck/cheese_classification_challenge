import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

class PixartGenerator:
    def __init__(self,use_cpu_offload=False):
        self.pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
    torch_dtype=torch.float16,
    use_safetensors=True,
)
        self.pipe.to(device)
        self.num_inference_steps=4
        self.guidance_scale = 0


    def generate(self, prompts):
       
        images= self.pipe(
            prompt=prompts, 
            num_inference_steps=self.num_inference_steps, 
            guidance_scale=self.guidance_scale
        ).images[0]
        
        return images