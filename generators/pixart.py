import torch
from diffusers import PixArtAlphaPipeline


device = "cuda" if torch.cuda.is_available() else "cpu"

class PixartGenerator:
    def __init__(self,use_cpu_offload=False):
        self.pipe =  PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
        self.pipe.to(device)
        self.num_inference_steps=4
        self.guidance_scale = 0


    def generate(self, prompts):
       
        images= self.pipe(
            prompt=prompts, 
            num_inference_steps=self.num_inference_steps, 
            guidance_scale=self.guidance_scale
        ).images
        
        return images