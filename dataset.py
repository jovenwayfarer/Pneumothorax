from torch.utils.data import DataLoader, Dataset
import pandas as pd
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import cv2
import os
import numpy as np
pd.options.mode.chained_assignment = None

class SIIMDataset(Dataset):
    def __init__(self, df, image_folder, mask_folder, size, mean, std, phase):
        self.df = df
        self.root_images = image_folder
        self.root_masks = mask_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)
        self.gb = self.df.groupby('ImageId')
        self.fnames = list(self.gb.groups.keys())

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        image_path = os.path.join(self.root_images, image_id + ".png")
        mask_path = os.path.join(self.root_masks, image_id + ".png")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)      
        mask = (mask > 0).astype(np.float32)

        
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                albu.HorizontalFlip(),
                albu.OneOf([
                    albu.RandomContrast(),
                    albu.RandomGamma(),
                    albu.RandomBrightness(),
                    ], p=0.3),
                albu.OneOf([
                    albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    albu.GridDistortion(),
                    albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                    ], p=0.3), 
                albu.ShiftScaleRotate(),
            ]
        )
    list_transforms.extend(
        [
            albu.Normalize(mean=mean, std=std, p=1),
            albu.Resize(size, size),
            ToTensorV2(),
        ]
    )

    list_trfms = albu.Compose(list_transforms)
    return list_trfms

def provider(
    fold,
    total_folds,
    image_folder,
    mask_folder,
    df_path,
    phase,
    size,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
    shuffle = False,
):
    df = pd.read_csv(df_path)

    df_with_mask = df[df["EncodedPixels"] != "-1"]
    df_with_mask['has_mask'] = 1
    df_without_mask = df[df["EncodedPixels"] == "-1"]
    df_without_mask['has_mask'] = 0
    df_without_mask_sampled = df_without_mask.sample(len(df_with_mask.drop_duplicates('ImageId')))
    df = pd.concat([df_with_mask, df_without_mask_sampled])
  
    
    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(
        df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%
    
    image_dataset = SIIMDataset(df, image_folder, mask_folder, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )
    return dataloader