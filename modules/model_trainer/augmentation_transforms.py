""" 
Data Augmentation Transforms for the model training pipeline.
"""
import albumentations as A
from settings import AppSettings

def get_transforms(config: AppSettings):

    transforms_train = A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(3,5)),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        # A.Cutout(max_h_size=int(config.IMAGE_SIZE * 0.375), max_w_size=int(config.IMAGE_SIZE * 0.375), num_holes=1, p=0.7),
        A.Normalize()
    ])

    transforms_val = A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize()
    ])

    return transforms_train, transforms_val