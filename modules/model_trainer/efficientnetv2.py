""" 
Efficientnet v2
"""
from typing import Any, Literal
import lightning as L
import torch
import torch.nn as nn
import torchmetrics
import lightning as L
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from settings import AppSettings

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EfficientNetV2(L.LightningModule):
    """ 
    Efficientnet v2
    """

    def __init__(
        self, 
        config: AppSettings,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = config.LEARNING_RATE
        self.patience = self.config.EARLY_STOPPING_PATIENCE
        self.model = efficientnet_v2_l(
            weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1,
        )
        self.model.classifier[1].out_features = num_classes
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _common_stage(self, batch: Any, stage: Literal["train", "val", "test"]) -> torch.Tensor:
        assert stage in ("train", "val", "test")
        data, label, label_onehot = batch
        output = self(data)
        loss = self.criterion(output, label)
        return loss, output, label
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, output, label = self._common_stage(batch, "train")
        self.log('train_loss', loss)
        self.train_acc(output, label)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, output, label = self._common_stage(batch, "val")
        self.val_acc(output, label)
        self.val_recall(output, label)
        self.val_precision(output, label)
        self.val_f1(output, label)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=True, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": self.val_acc, "val_recall": self.val_recall, "val_precision": self.val_precision, "val_f1": self.val_f1}
    
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, output, label = self._common_stage(batch, "test")
        self.test_acc(output, label)
        self.test_recall(output, label)
        self.test_precision(output, label)
        self.test_f1(output, label)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)
        self.log('test_recall', self.test_recall, on_step=True, on_epoch=True)
        self.log('test_precision', self.test_precision, on_step=True, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=True, on_epoch=True)
        return {"test_loss": loss, "test_acc": self.test_acc, "test_recall": self.test_recall, "test_precision": self.test_precision, "test_f1": self.test_f1}
    
    
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
