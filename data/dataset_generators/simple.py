from .base import DatasetGenerator


class SimplePromptsDatasetGenerator(DatasetGenerator):
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
        prompts = {}
        adding=["box","cartons","plates","many elements"]  #ajout
        for label in labels_names:
            prompts[label] = []
            for add in adding:
                prompts[label].append(
                {
                    "prompt": f"An image of a {add} of {label} cheese",  #ajout
                    "num_images": self.num_images_per_label,
                }
            )
                
        return prompts
