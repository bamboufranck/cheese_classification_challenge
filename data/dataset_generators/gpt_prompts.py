import torch
from .base import DatasetGenerator
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torchvision.transforms as transforms




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

        
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")
      
        map_images={}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for label in labels_names:
            map_images[label]=[]
            
        model.to(device)
        to_pil = transforms.ToPILImage()


        prompts = {}

        for label in labels_names:
            prompts[label]=[]
            prompt_text = f"describe an image of {label} cheese in differents places or in differents dishes with differents aliments:"
            inputs = tokenizer(prompt_text , return_tensors='pt', padding=True, truncation=True, max_length=30)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

                
            outputs = model.generate(input_ids=input_ids,attention_mask=attention_mask,pad_token_id=tokenizer.pad_token_id,max_new_tokens=50)
            
            generated_texts = tokenizer.decode(outputs[0], skip_special_tokens=True) 

            prompts[label].append({"prompt": generated_texts, "num_images": self.num_images_per_label})

            prompts[label].append({
                    "prompt": f"an image of {label} cheese",
                    "num_images": self.num_images_per_label,
                })
            
        for i,batch in enumerate(val_data):
            image, label = batch
            valeur_label = label[0].item()
            image = image.squeeze(0)
            map_images[maping[valeur_label]].append(image)
            image = to_pil(image)

        del model
        torch.cuda.empty_cache() 

        return prompts,map_images
