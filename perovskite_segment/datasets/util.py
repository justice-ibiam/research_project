from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
import torch
from torch.utils.data import WeightedRandomSampler
import os
from .solarpark.dataset import SolarPark



def get_dataset(cfg, split):
    if cfg.dataset.name == 'solarpark':
        image_dir = os.path.join(cfg.dataset.path, split, "images")
        mask_dir = os.path.join(cfg.dataset.path, split, "masks")
        print(mask_dir)
        dataset = SolarPark(cfg, image_dir, mask_dir, split)
        shuffle = True if split in ['train',"valid"] else False
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=shuffle, num_workers=0)
        return dataset, dataloader
    else:
        raise ValueError(f'Dataset "{cfg.dataset.name}" unknown')