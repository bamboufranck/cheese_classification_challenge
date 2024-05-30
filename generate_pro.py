import torch
import wandb
import hydra
from tqdm import tqdm


@hydra.main(config_path="configs/generate", config_name="config")
def generate(cfg):
    dataset_generator = hydra.utils.instantiate(cfg.dataset_generator)
    valmodule = hydra.utils.instantiate(cfg.get_val)


    fromages = [
    "OSSAU- IRATY",
    "MIMOLETTE",
    "MAROILLES",
    "GRUYÈRE",
    "MOTHAIS",
    "VACHERIN",
    "MOZZARELLA",
    "TÊTE DE MOINES",
    "FROMAGE FRAIS"
]


    #labels=["FETA","TOMME DE VACHE","VACHERIN","TÊTE DE MOINES","CHABICHOU","EMMENTAL","FROMAGE FRAIS","GRUYÈRE","MOTHAIS","MOZZARELLA","OSSAU- IRATY","REBLOCHON","PECORINO","SAINT- FÉLICIEN"]

    #labels1=["BRIE DE MELUN", "CAMEMBERT","EPOISSES","FOURME D’AMBERT","RACLETTE", "MORBIER","SAINT-NECTAIRE", "POULIGNY SAINT- PIERRE","ROQUEFORT","COMTÉ","CHÈVRE","PECORINO","NEUFCHATEL","CHEDDAR","BÛCHETTE DE CHÈVRE","PARMESAN","SAINT- FÉLICIEN"]


    labels1=["MUNSTER","NEUFCHATEL","GRUYÈRE","CABECOU"]  # pedrix
    labels2=["TÊTE DE MOINES","MOZZARELLA"]    #oriol
    labels3=["CHÈVRE","MONT D’OR"]
    with open(cfg.labels_file, "r") as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]
   
    val_loaders,maping  = valmodule.val_real_dataloader()

    # PEDRIX je génère labels1 
    #Quiscale je génère labels/labels1
    # LORIOL je génère tout ce qui n'est pas dans not car j'ai déja fait le reste





    

    for label in labels:
        if label in labels3: 
            dataset_generator.generate(label.strip(),labels,val_loaders,maping)

    """

    for label in fromages:
        dataset_generator.generate(label.strip(),labels,val_loaders,maping)

      """

    








    #label="MONT D’OR"
    #dataset_generator.generate(label.strip(),labels,val_loaders,maping)
        
    


if __name__ == "__main__":
    generate()
