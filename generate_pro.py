import torch
import wandb
import hydra
from tqdm import tqdm


@hydra.main(config_path="configs/generate", config_name="config")
def generate(cfg):
    dataset_generator = hydra.utils.instantiate(cfg.dataset_generator)
    valmodule = hydra.utils.instantiate(cfg.get_val)


    labels=["FETA","TOMME DE VACHE","VACHERIN","TÊTE DE MOINES","CHABICHOU","EMMENTAL","FROMAGE FRAIS","GRUYÈRE","MOTHAIS","MOZZARELLA","OSSAU- IRATY","REBLOCHON","PECORINO","SAINT- FÉLICIEN"]

    #with open(cfg.labels_file, "r") as f:
        #labels = f.readlines()
        #labels = [label.strip() for label in labels]
   
    val_loaders,maping  = valmodule.val_real_dataloader()

    for label in labels:
        dataset_generator.generate(label.strip(),labels,val_loaders,maping)


if __name__ == "__main__":
    generate()
