from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from hydra.utils import instantiate
import torch


class Get_val:
    def __init__(
        self,
        real_images_val_path,
        test_images_path,
        batch_size,
        num_workers,
    ):
        transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            ])
        
        self.real_images_val_dataset = ImageFolder(
            real_images_val_path, transform=transformations
        )
        self.batch_size = 1
        self.num_workers = num_workers
        self.idx_to_class = {v: k for k, v in self.real_images_val_dataset.class_to_idx.items()}
        self.test_images= ImageFolder(
            test_images_path, transform=transformations
        )
   
    def val_real_dataloader(self):
        return DataLoader(
                self.real_images_val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            ), self.idx_to_class,DataLoader(
                self.test_images,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        
