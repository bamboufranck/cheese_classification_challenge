from transformers import GPT2Tokenizer, GPT2LMHeadModel
from .base import DatasetGenerator

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

class GptPromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=200,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label

    def create_prompts(self, labels_names):
        descriptions = {}
        adding = ["box", "cartons", "plates", "many"]

        for label in labels_names:
            descriptions[label] = []
            for add in adding:
                prompt = f"An image of a {add} of {label} cheese"

               
                inputs = tokenizer.encode_plus(
                    prompt, return_tensors='pt', padding='max_length', max_length=100, truncation=True
                )
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                     max_new_tokens=30
                )
                description = tokenizer.decode(outputs[0], skip_special_tokens=True)
                descriptions[label].append(description)

        prompts = {}
        for label in labels_names:
            prompts[label] = []
            elements = descriptions[label]
            for e in elements:
                prompts[label].append({
                    "prompt": e,
                    "num_images": self.num_images_per_label,
                })

        return prompts
