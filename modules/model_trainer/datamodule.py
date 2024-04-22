""" 
Data Module for the model training pipeline.
"""
import lightning as L
import torch 
from torch.utils.data import DataLoader
from modules.model_trainer.dataset import ISIC_2018_Dataset
from settings import AppSettings
from modules.model_trainer.augmentation_transforms import get_transforms
from torchvision import transforms


data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop((299, 299)),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((308, 308)),
                                   transforms.CenterCrop((299, 299)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]),
        "test": transforms.Compose([transforms.Resize((308, 308)),
                                   transforms.CenterCrop((299, 299)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ])
        }


class ImageClassificationDataModule(L.LightningDataModule):
    """Image Classification Data Module for the model training pipeline."""
    
    def __init__(
        self,
        config: AppSettings,
        train_img_folder: str,
        train_ground_truth_file: str,
        val_img_folder: str,
        val_ground_truth_file: str,
        test_img_folder: str,
        test_ground_truth_file: str,
    ) -> None:
        super().__init__()        
        self.batch_size = config.BATCH_SIZE
        self.num_workers = config.NUM_WORKERS
        # Train Data
        self.train_img_folder = train_img_folder
        self.train_ground_truth_file = train_ground_truth_file
        # Validation Data
        self.val_img_folder = val_img_folder
        self.val_ground_truth_file = val_ground_truth_file
        # Test Data
        self.test_img_folder = test_img_folder
        self.test_ground_truth_file = test_ground_truth_file
        # Set transforms
        # self.transforms_train, self.transforms_val = data_transform["train"], data_transform["val"]
        self.transforms_train, self.transforms_val = get_transforms(config)
        # Setup the data
        self.setup()
        self.save_hyperparameters()
        
    def setup(self, stage: str = None) -> None:
        """Setup the data for the model training pipeline."""
        if stage == 'fit' or stage is None:
            self.train_dataset = ISIC_2018_Dataset(
                img_folder=self.train_img_folder,
                ground_truth_file=self.train_ground_truth_file,
                transform=self.transforms_train,
            )
            self.val_dataset = ISIC_2018_Dataset(
                img_folder=self.val_img_folder,
                ground_truth_file=self.val_ground_truth_file,
                transform=self.transforms_val,
            )
        if stage == 'test':
            self.test_dataset = ISIC_2018_Dataset(
                img_folder=self.test_img_folder,
                ground_truth_file=self.test_ground_truth_file,
                transform=self.transforms_val,
            )
    
    # def prepare_data(self) -> None:
    #     """Has to be implemented to avoid object has no attribute 'prepare_data_per_node' error."""
    #     super().prepare_data()

    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    