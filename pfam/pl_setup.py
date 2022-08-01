import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

class Light(pl.LightningModule):
    def __init__(self, protcnn, milestones, gamma, lr, weight_decay):
        super().__init__()
        self.protcnn = protcnn
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.milestones = milestones
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.protcnn(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1) 
        acc = self.valid_acc(pred, y)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

        return acc
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
