import torch
import wandb
import hydra
from tqdm import tqdm


@hydra.main(config_path="configs/generate", config_name="config")
def generate(cfg):
    dataset_generator = hydra.utils.instantiate(cfg.dataset_generator)
    valmodule = hydra.utils.instantiate(cfg.get_val)

    label="MOZZARELLA"

   

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    print("dim",backbone.num_features)


    val_loaders,maping  = valmodule.val_real_dataloader()

   
    dataset_generator.generate(label.strip(),val_loaders,maping)


if __name__ == "__main__":
    generate()
