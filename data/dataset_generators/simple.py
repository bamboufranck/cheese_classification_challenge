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

    def create_prompts(self, labels_names,val_data,maping):
        prompts = {}
        situations = ["kitchen", "dishes", "table", "boxes","with persons","with a knife and meat","with a piece of cake","with a piece of bread","with a wooden cutting board","a yellow plastic container filled with this cheese","a table topped with lots of different types of food"]
        for label in labels_names:
            prompts[label] = []
            for situation in situations:
                prompts[label].append(
                {
                    "prompt": f"An image of a {label} cheese in {situation}",
                    "num_images": self.num_images_per_label,
                }
            )
        return prompts
