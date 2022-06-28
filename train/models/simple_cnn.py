# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import madgrad
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from torchvision import models

class SimpleCNNModel(LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super(SimpleCNNModel, self).__init__()

        self.cfg = cfg
        # self.save_hyperparameters(cfg)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        ## Input (240x320)
        # self.fc0 = nn.Linear(70224, 5000)
        # self.relu0 = nn.ReLU()
        # self.fc = nn.Linear(5000, 2500)
        # self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(2500, 500)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(500, cfg.model.n_classes)

        ## Input (64*64)
        self.fc0 = nn.Linear(2704, 400)
        self.relu0 = nn.ReLU()
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, cfg.model.n_classes)


        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, frame):
        out = self.layer1(frame)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc0(out)
        out = self.relu0(out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out

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
        # optimF = torch.optim.Adam
        optimF = torch.optim.SGD
        # optimF = madgrad.MADGRAD
        optimizer = optimF(self.parameters(), lr=self.cfg.optimizer.lr)

        return {
            "optimizer": optimizer,
        }
