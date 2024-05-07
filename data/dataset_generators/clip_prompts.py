import torch
from .base import DatasetGenerator
from clip_interrogator import Interrogator
from PIL import Image

ci = Interrogator()

class ClipPromptsDatasetGenerator(DatasetGenerator):
    def __init__(
        self,
        generator,
        batch_size=1,
        output_dir="dataset/train",
        num_images_per_label=10,
    ):
        super().__init__(generator, batch_size, output_dir)
        self.num_images_per_label = num_images_per_label


    def create_prompts(self, labels_names,val_data):
        prompts = {}
        for label in labels_names:
            prompts[label]=[]

        for i,batch in enumerate(val_data):
            image, label = batch
            prompts[label].append(
                {
                    "prompt": ci.interrogate(image),
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts