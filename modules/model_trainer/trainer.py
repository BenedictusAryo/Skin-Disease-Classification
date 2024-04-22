""" 
Pytorch Lightning Trainer Module
"""
from pathlib import Path
import lightning as L
from settings import AppSettings
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger


wandb_logger = WandbLogger(log_model='all', project='SkinDiseaseClassification_ISIC_2018')
tensorboard_logger = TensorBoardLogger(save_dir='models')

class TunerModule:
    """Tuner Module."""
        
    def __new__(cls, trainer:L.Trainer) -> L.pytorch.tuner.Tuner:
        return L.pytorch.tuner.Tuner(trainer)
    

class TrainerModule:
    """Trainer Module."""
    
    def __init__(
        self,
        config: AppSettings,
    ) -> None:
        self.trainer = L.Trainer(
            accelerator='auto',
            devices=1,
            max_epochs=config.MAX_EPOCHS,
            log_every_n_steps=config.LOG_EVERY_N_STEPS,
            default_root_dir=Path(config.MODEL_DIR),
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=config.EARLY_STOPPING_PATIENCE,
                    mode='min'
                ),
                ModelCheckpoint(
                    filename="best_model_{epoch}-{val_loss:.2f}-{val_acc:.2f}",
                    monitor='val_loss',
                    mode='min'
                )
            ],
            logger=[wandb_logger, tensorboard_logger],
        )
        
        
    # def __new__(cls, config: AppSettings) -> L.Trainer:
    #     return cls(config).trainer
            