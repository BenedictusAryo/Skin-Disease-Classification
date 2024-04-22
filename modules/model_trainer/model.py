"""
Model Definition and Model method
"""
from typing import Any, Literal, Optional, Union
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchsummary import summary
from PIL import Image
from modules.model_trainer.fixcaps import FixCapsNet
from settings import AppSettings


class ImageClassificationModel(L.LightningModule):
    """
    Image Classification Model
    """
    
    def __init__(
        self,
        config: AppSettings,
        num_classes: int,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.patience = self.config.EARLY_STOPPING_PATIENCE
        self.model = FixCapsNet(
            conv_inputs=3,
            conv_outputs=128,
            primary_units=8,
            primary_unit_size=16 * 6 * 6,
            output_unit_size=16,
            num_classes=num_classes,
            init_weights=True,
            mode='DS'
        )
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _common_stage(self, batch: Any, stage: Literal["train", "val", "test"]) -> torch.Tensor:
        assert stage in ("train", "val", "test")
        data, label, label_onehot = batch
        output = self(data)
        loss = self.model.loss(output, label_onehot)
        v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True)) 
        preds = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze()
        return preds, label, loss
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds, labels, loss = self._common_stage(batch, "train")
        self.log("train_loss", loss)
        self.train_acc(preds, labels)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds, labels, loss = self._common_stage(batch, "val")
        self.log("val_loss", loss)
        self.val_acc(preds, labels)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_recall(preds, labels)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.val_precision(preds, labels)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.val_f1(preds, labels)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_preds":preds, "val_labels":labels, "val_acc": self.val_acc, "val_recall": self.val_recall, "val_precision": self.val_precision, "val_f1": self.val_f1}
    
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds, labels, loss = self._common_stage(batch, "test")
        self.log("test_loss", loss)
        self.test_acc(preds, labels)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.test_recall(preds, labels)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.test_precision(preds, labels)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.test_f1(preds, labels)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_preds":preds, "test_labels":labels, "test_acc": self.test_acc, "test_recall": self.test_recall, "test_precision": self.test_precision, "test_f1": self.test_f1}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=0.1, 
                    patience=self.patience,
                ),
                "monitor": "val_loss"
            },
        }
    
    def print_model_summary(self) -> None:
        print(summary(self.model, (3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)))
