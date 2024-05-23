import torch
import wandb
import hydra
from tqdm import tqdm


@hydra.main(config_path="configs/generate", config_name="config")
def generate(cfg):
    dataset_generator = hydra.utils.instantiate(cfg.dataset_generator)
    valmodule = hydra.utils.instantiate(cfg.get_val)

    label="MOTHAIS"

    val_loaders,maping  = valmodule.val_real_dataloader()

    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    #print(labels)
    #print("mon label",label.strip())

    dataset_generator.generate(label.strip(),val_loaders,maping)


if __name__ == "__main__":
    generate()
