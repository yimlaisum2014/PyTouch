# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import madgrad
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from torchvision import models


class TouchDetectionModel(LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.model = models.resnet18(pretrained=cfg.model.pre_trained)
        # self.model = models.resnet101(pretrained=cfg.model.pre_trained)
        # self.model = models.vgg16(pretrained=cfg.model.pre_trained)
        
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Linear(self.model.fc.in_features, cfg.model.n_classes)
        # self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=cfg.model.n_classes)

        self.criterion = nn.CrossEntropyLoss()


        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, frame):
        return self.model(frame)

    def training_step(self, batch, batch_idx):
        images, targets, sn = batch
        output = self.forward(images)
        train_loss = self.criterion(output, targets)

        self.train_accuracy(output.argmax(dim=1), targets)
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        images, targets, sn = batch
        output = self.forward(images)
        val_loss = self.criterion(output, targets)

        self.val_accuracy(output.argmax(dim=1), targets)

        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        #optimF = torch.optim.Adam
        optimF = torch.optim.SGD
        #optimF = madgrad.MADGRAD
        optimizer = optimF(self.parameters(), lr=self.cfg.optimizer.lr)

        return {
            "optimizer": optimizer,
        }
