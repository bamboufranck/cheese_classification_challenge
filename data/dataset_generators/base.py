from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from torch import optim

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

    def generate(self, labels_names,val_data,maping):

        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")   # Ajout
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   # Ajout

        labels_names_with_cheese = [name + " cheese" for name in labels_names]
        text_inputs = processor(text=labels_names_with_cheese, return_tensors="pt", padding=True) # Ajout

        

        # ensuite ici fine tune mon générateur avec val_data ou tout simplement utilise ses images 
        # pour mieux generer

        # ou encore utiliser ca pour generer de meilleur prompt avec clip interrogator par exemple 

        self.fine_tune(val_data,maping)

        labels_prompts = self.create_prompts(labels_names,val_data,maping)
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
                    predicted_category = labels_names_with_cheese[predicted_index]           # Ajout

                    if(predicted_category==label+ " cheese"):                              # Ajout
                        self.save_images(images, label, image_id_0)            
                        image_id_0 += len(images)                               
                        pbar.update(1)
                        
                    """""
                    self.save_images(images, label, image_id_0)            
                    image_id_0 += len(images)                               
                    pbar.update(1)
                    """""

                    
                pbar.close()

        
        del model
        torch.cuda.empty_cache()
        

    def fine_tune(self,val_data,maping):

        epochs=3
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")   # Ajout
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")   # Ajout
        to_pil = transforms.ToPILImage()
        optimizer = optim.Adam(self.generator.unet.parameters(), lr=1e-3)
        target_similarity = 1.0

        print("start of fine tuning")

        for epoch in tqdm(range(epochs)):
            for i,batch in enumerate(val_data):

                optimizer.zero_grad()
                image, label = batch
                image = image.squeeze(0)
                image = to_pil(image)
                valeur_label = label[0].item()
                prompt=f"An image of {maping[valeur_label]} cheese"

                generate_image = self.generator.generate(prompt)
                generate_image_input = processor(images=generate_image, return_tensors="pt")  # Ajout
                image_input=  processor(images=image, return_tensors="pt")

                with torch.no_grad():
                    image_features = model.get_image_features(**image_input) # Ajout
                    generate_image_features = model.get_image_features(**generate_image_input)  # Ajout


                similarity = torch.nn.functional.cosine_similarity(generate_image_features, image_features, dim=1)
                loss = torch.abs(similarity - target_similarity)

                loss.backward()
                optimizer.step()


        del model
        torch.cuda.empty_cache()
        print("end of fine tuning")
    
    def create_prompts(self, labels_names,val_data,maping):
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
