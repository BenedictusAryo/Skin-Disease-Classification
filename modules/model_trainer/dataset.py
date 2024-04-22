""" 
Dataset Loader for Image Classification
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
# from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader


ISIC_2018_CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


class ISIC_2018_Dataset(datasets.ImageFolder):
    """ 
    ISIC 2018 Dataset that contains images and labels
    """
    
    def __init__(
        self, 
        img_folder: str,
        ground_truth_file: str,
        image_ext: str = ".jpg",
        transform: Callable[..., Any] | None = None
    ) -> None:
        if not Path(img_folder).exists():
            raise FileNotFoundError(f"Image Folder {img_folder} not found. Exiting...")
        if not Path(ground_truth_file).exists():
            raise FileNotFoundError(f"Ground Truth File {ground_truth_file} not found. Exiting...")
        self.img_folder = Path(img_folder)
        self.image_files = list(self.img_folder.glob(f"*{image_ext}"))
        self.classes = ISIC_2018_CLASSES
        self.num_classes = len(self.classes)
        self.ground_truth = (
            pd.read_csv(ground_truth_file)
            .set_index("image")
            # .melt(id_vars="image", var_name="class", value_name="label")
            # .loc[lambda x: x['label'] == 1]
            # .set_index("image")["class"]
        )
        if len(self.ground_truth) != len(self.image_files):
            raise ValueError("Mismatch in number of images and ground truth labels")
        self.class_to_idx = {cls: idx for idx, cls in enumerate(ISIC_2018_CLASSES)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_files[idx])
        if self.transform:
            img = self.transform(img)
        # img = cv2.imread(
        #     str(self.image_files[idx])
        # )[..., ::-1] # BGR to RGB
        # if self.transform:
        #     img = self.transform(image=img)['image']
        #     img = torch.from_numpy(img).permute(2, 0, 1) # HWC to CHW
        label_array = self.ground_truth.loc[self.image_files[idx].stem].values
        label = np.argmax(label_array)
        label_onehot = torch.from_numpy(label_array.reshape(-1, self.num_classes)).float()
        return img, label, label_onehot
        