import torch
from .base import DatasetGenerator
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialisation du modèle et du tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Configuration de padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Configuration de l'appareil
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class GptPromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
         batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=4,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label

    def create_prompts(self, labels_names,val_data,maping):
        descriptions = {}
        situations = ["kitchen", "dishes", "table", "boxes","with persons","with a knife and meat"]

        for label in labels_names:
            prompts = [f"description an image of {label} cheese in  {situation}" for situation in situations]
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=30)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():  
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=20
                )
            
            generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            descriptions[label] = generated_texts

        prompts = {
            label: [{"prompt": description, "num_images": self.num_images_per_label} for description in descriptions[label]]
            for label in labels_names
        }

        return prompts
