import torch
import torch.nn as nn
import timm

from pytorch_lightning import LightningModule
import torchmetrics
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

class TimmClassifier(LightningModule):
    def __init__(self, num_classes, backbone, optimizer, scheduler):
        super().__init__()
        
        self.accuracy = torchmetrics.Accuracy(task = 'multiclass', num_classes=num_classes)
        self.f1_macro = torchmetrics.F1Score(task = 'multiclass', num_classes=num_classes, average='macro')
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model = instantiate(backbone)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        logit = self.model(x)
        return logit
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logit = self(images)

        loss = self.criterion(logit, labels)
        acc = self.accuracy(logit, labels)

        self.log("train_loss", loss, on_step= False, on_epoch = True, logger = True, prog_bar=True)
        self.log("train_acc", acc, on_step= False, on_epoch = True, logger = True, prog_bar=False)

        return loss
    

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logit = self(images)

        loss = self.criterion(logit, labels)
        acc = self.accuracy(logit, labels)

        preds = self.softmax(logit)
        f1 = self.f1_macro(preds, labels)

        self.log("valid_loss", loss, on_step= False, on_epoch = True, logger = True, prog_bar=True)
        self.log("valid_acc", acc, on_step= False, on_epoch = True, logger = True, prog_bar=False)
        self.log("valid_f1", f1, on_step= False, on_epoch = True, logger = True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, self.parameters())

        if self.scheduler is None or (isinstance(self.scheduler, DictConfig) and self.share_memory._is_none()):
            return optimizer
        
        scheduler = instantiate(self.scheduler, optimizer)
        
        return [optimizer], [scheduler]
    

    def predict_step(self, batch, batch_idx):
        images, labels = batch
        logit = self(images)

        return logit