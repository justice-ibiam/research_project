from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SolarPark(Dataset):

    def __init__(
        self,
        cfg,
        image_dir,
        mask_dir,
        split="train",
        transform=None
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.split = split
        self.mean = list(cfg.dataset.mean)
        self.std = list(cfg.dataset.std)

        # self.images = sorted(
        #     list(self.image_dir.glob("*"))
        # )
        self.images = []

        for image_path in sorted(self.image_dir.glob("*.jpg")):
            mask_path = self.mask_dir / (image_path.stem + ".png")

            if mask_path.exists():
                self.images.append(image_path)
            else:
                print(f"Warning: No mask for {image_path.name}. Skipping.")

        # self.transform = transform
        if self.split=="train":
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),

                A.Rotate(limit=10, p=0.5),

                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),

                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.3,
                ),

                A.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),

                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(
                    mean=cfg.dataset.mean,
                    std=cfg.dataset.std,
                ),

                ToTensorV2(),
            ])


    def __len__(self):

        return len(self.images)


    def __getitem__(self, idx):
        # image path
        image_path = self.images[idx]

        # corresponding mask path
        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        # read image
        image = cv2.imread(
            str(image_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        image = cv2.resize(
            image,
            (320,240),
            interpolation=cv2.INTER_LINEAR
        )

        # read mask
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.resize(
            mask,
            (320,240),
            interpolation=cv2.INTER_NEAREST
        )

        # binary mask
        mask = (mask > 0).astype("float32")


        # augmentation
        # print(image.shape)
        if self.transform:

            augmented = self.transform(
                image=image,
                mask=mask
            )

            image = augmented["image"]
            mask = augmented["mask"]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        # image = image / 255.0


        



        # mask = mask.unsqueeze(0)

        # print(image.shape)
        # print(mask.shape)

        return image, mask