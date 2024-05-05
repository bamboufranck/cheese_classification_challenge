from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class DatasetGenerator:
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
    ):
        """
        Args:
            generator: image generator object
            batch_size: Number of images to generate per batch. Make sure to max out your GPU VRAM for maximum efficiency
            output_dir: Directory where the generated images will be saved
        """
        self.generator = generator
        self.batch_size = batch_size
        self.output_dir = output_dir

    def generate(self, labels_names):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")   # Ajout
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   # Ajout
        text_inputs = processor(text=labels_names, return_tensors="pt", padding=True) # Ajout

        labels_prompts = self.create_prompts(labels_names)
        for label, label_prompts in labels_prompts.items():
            image_id_0 = 0
            for prompt_metadata in label_prompts:
                num_images_per_prompt = prompt_metadata["num_images"]
                prompt = [prompt_metadata["prompt"]] * num_images_per_prompt
                pbar = tqdm(range(0, num_images_per_prompt, self.batch_size))
                pbar.set_description(
                    f"Generating images for prompt: {prompt_metadata['prompt']}"
                )
                for i in range(0, num_images_per_prompt, self.batch_size):
                    batch = prompt[i : i + self.batch_size]
                    images = self.generator.generate(batch)

                    image_input = processor(images=images, return_tensors="pt")  # Ajout

                    with torch.no_grad():
                         image_features = model.get_image_features(**image_input) # Ajout
                         text_features = model.get_text_features(**text_inputs)  # Ajout
                    
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Ajout
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Ajout

                    similarities = torch.matmul(image_features, text_features.T)  # Ajout
                    predicted_index = similarities.argmax().item()               # Ajout
                    predicted_category = labels_names[predicted_index]           # Ajout

                    if(predicted_category==label):                              # Ajout
                        self.save_images(images, label, image_id_0)            
                        image_id_0 += len(images)                               
                        pbar.update(1)
                pbar.close()

    def create_prompts(self, labels_names):
        """
        Prompts should be a dictionary with the following structure:
        {
            label_0: [
                {
                    "prompt": "Prompt for label_0",
                    "num_images": 100
                },
                {
                    "prompt": "Another prompt for label_0",
                    "num_images": 200
                }
            ],
            label_1: [
                {
                    "prompt": "Prompt for label_1",
                    "num_images": 100
                }
            ]
        }
        """
        return NotImplementedError

    def save_images(self, images, label, image_id_0):
        output_path = Path(self.output_dir) / label
        output_path.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            image.save(output_path / f"{str(image_id_0 + i).zfill(6)}.jpg")
