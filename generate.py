import torch
import wandb
import hydra
from tqdm import tqdm


@hydra.main(config_path="configs/generate", config_name="config")
def generate(cfg):
    dataset_generator = hydra.utils.instantiate(cfg.dataset_generator)
    valmodule = hydra.utils.instantiate(cfg.get_val)

    val_loaders = valmodule.val_real_dataloader()

    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    dataset_generator.generate(labels,val_loaders)


if __name__ == "__main__":
    generate()
