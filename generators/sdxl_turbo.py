from diffusers import AutoPipelineForText2Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class SDXLTurboGenerator:
    def __init__(self,use_cpu_offload=False):
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
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