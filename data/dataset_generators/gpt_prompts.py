import torch
from .base import DatasetGenerator
from transformers import T5Tokenizer, T5ForConditionalGeneration


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

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
        prompts = {}
        situations = ["kitchen", "dishes", "table", "boxes","with persons","with a knife and meat","with a piece of cake","with a piece of bread","with a wooden cutting board","a yellow plastic container filled with this cheese","a table topped with lots of different types of food"]

        for label in labels_names:
            prompts[label]=[]
            for situation in situations:
                prompts = f"describe an image of {label} cheese in  {situation}:"
                inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=30)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                
                outputs = model.generate(input_ids=input_ids,attention_mask=attention_mask,pad_token_id=tokenizer.pad_token_id,max_new_tokens=20)
            
                generated_texts = tokenizer.decode(outputs[0], skip_special_tokens=True) 

                prompts[label].append({"prompt": generated_texts, "num_images": self.num_images_per_label})

           

        return prompts
