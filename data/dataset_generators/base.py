from pathlib import Path
from tqdm import tqdm
import torch
import random
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from torch import optim
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

# Initialisation du tokenizer et du modèle
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Préparer l'entrée
inputs = tokenizer("Hello, my name is ChatGPT.", return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Passer l'entrée au modèle
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

# Récupérer les états cachés (représentations encodées)
encoded_representations = outputs.last_hidden_state

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
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")    # Ajout

        
        

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

                    text_inputs = processor(text="A {label} cheese", return_tensors="pt", padding=True)
                    image_input = processor(images=images, return_tensors="pt")  # Ajout

                    with torch.no_grad():
                         image_features = model.get_image_features(**image_input) # Ajout
                         text_features = model.get_text_features(**text_inputs)  # Ajout
                    
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True) # Ajout
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Ajout

                    similarities = torch.matmul(image_features, text_features.T)  # Ajout
                    score_similarity =  similarities.squeeze().item()               # Ajout
                               

                    if(score_similarity>=0.30):                              
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


        def get_random_batch_from_loader(dataloader):
            total_batches = len(dataloader)
            random_batch_index = random.randint(0, total_batches - 1)
            for i, batch in enumerate(dataloader):
                if i == random_batch_index:
                    return batch
                

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
            device, torch.float16
        )
        scheduler_config = {
    "timestep_respacing": "25",  # Exemple de paramètre, ajustez selon les besoins réels
    "betas": (0.9, 0.999)        # Autre exemple de paramètre pour l'optimiseur
}
       

        
        noise_scheduler = EulerDiscreteScheduler.from_config(scheduler_config, timestep_spacing="trailing")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)

        epochs=3
        prior_loss_weight=1
        optimizer = optim.Adam(unet.parameters(), lr=1e-4)
        prompt_general="a cheese"
        text_inputs_general=tokenizer(prompt_general,truncation=True,padding="max_length",max_length=20,return_tensors="pt").to(device)

        
        
       

        print("start of fine tuning")
        for epoch in tqdm(range(epochs)):
            unet.train()
            for i,batch in enumerate(val_data):
                bibi=get_random_batch_from_loader(val_data)
                example = {}
                optimizer.zero_grad()
                image,label = batch
                valeur_label = label[0].item()

                prompt=f"A  {maping[valeur_label]} cheese"
                example["instance_images"]=image
                text_inputs=tokenizer(prompt,truncation=True,padding="max_length",max_length=20,return_tensors="pt").to(device)
                example["instance_prompt_ids"] = text_inputs.input_ids
                example["instance_attention_mask"] = text_inputs.attention_mask

                class_image,label=bibi
                classe_instance=class_image
                class_prompt_ids=text_inputs_general.input_ids
                class_attention_mask= text_inputs_general.attention_mask

                example["instance_prompt_ids"] = torch.cat([text_inputs.input_ids, class_prompt_ids], dim=0)
                example["instance_attention_mask"] = torch.cat([text_inputs.attention_mask, class_attention_mask], dim=0)
                example["instance_images"] = torch.cat([image, classe_instance], dim=0)

               


                pixel_values =example["instance_images"]
                pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
                input_ids = example["instance_prompt_ids"]
                attention_mask = example["instance_attention_mask"]

               

                batch = {"input_ids": input_ids, "pixel_values": pixel_values, "attention_mask":attention_mask,}

                pixel_values = batch["pixel_values"].to(device)
                model_input = pixel_values


                noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

               

                noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)  # Ajout de bruit
                target = noise 

            

                encoder_hidden=model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state

                print("Noisy images shape:", noisy_images.shape)
                #print("Encoder hidden state shape:", encoder_hidden.shape)
                print("Timesteps shape:", timesteps.shape)

                text_embeddings = encoder_hidden[:, 0, :]

                added_cond_kwargs = {"text_embeds": text_embeddings,"time_ids": timesteps}
                
                
                noisy_images = noisy_images.to(device).half()
                encoder_hidden=encoder_hidden.half()
                encoder_hidden=encoder_hidden.to(device)
                model_pred = unet(noisy_images, timesteps, encoder_hidden, return_dict=False,added_cond_kwargs=added_cond_kwargs)[0]

                print("Azoa")

                loss = F.mse_loss(model_pred, target)

                print("Azoa")

                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)

                print("azoa")
                   
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                loss = loss + prior_loss_weight * prior_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()





        self.generator.update(unet)

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
