"""
Train Model
"""
from modules.model_trainer.datamodule import ImageClassificationDataModule
from modules.model_trainer.dataset import ISIC_2018_CLASSES, ISIC_2018_Dataset
from modules.model_trainer.model import ImageClassificationModel
from modules.model_trainer.trainer import TrainerModule, TunerModule
from modules.model_trainer.augmentation_transforms import get_transforms
from settings import AppSettings


# Get config
config = AppSettings()

# Get Transforms
transforms_train, transforms_val = get_transforms(config)

# Load Dataset
# Create Data Module
data_module = ImageClassificationDataModule(
    config=config,
    train_img_folder=r"data/isic/2018/ISIC2018_Task3_Training_Input",
    train_ground_truth_file=r"data/isic/2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
    val_img_folder=r"data/isic/2018/ISIC2018_Task3_Validation_Input",
    val_ground_truth_file=r"data/isic/2018/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv",
    test_img_folder=r"data/isic/2018/ISIC2018_Task3_Test_Input",
    test_ground_truth_file=r"data/isic/2018/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv",
)

# Model
model = ImageClassificationModel(
    config=config,
    num_classes=len(ISIC_2018_CLASSES),
    learning_rate=config.LEARNING_RATE,
)

# Trainer
trainer = TrainerModule(
    config=config,    
).trainer

# Run the model
if __name__ == "__main__":
    # Model Summary
    print("Model Summary")
    model.print_model_summary()
    # Tune to fine the best hyperparameters
    tuner = TunerModule(trainer)
    # Auto-scale batch size with binary search
    tuner.scale_batch_size(model=model, datamodule=data_module, mode="binsearch")

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    test_result = trainer.test(ckpt_path="best", datamodule=data_module)

    print(f"Test Result: {test_result}")
    print("Done!")
