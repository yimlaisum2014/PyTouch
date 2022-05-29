# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import sys
import os.path as osp

import hydra
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold
from copy import deepcopy

from typing import Any, Dict, List, Optional, Type
from pytouch.datasets.digit import DigitFolder
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn import functional as F
from pytorch_lightning.core import LightningModule, LightningDataModule
from torchmetrics.classification.accuracy import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from datamodules.touch_detect import BaseKFoldDataModule
_log = logging.getLogger(__name__)

class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.test_acc = Accuracy()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.nll_loss(logits, batch[1])
        self.test_acc(logits, batch[1])
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)



class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

@hydra.main(config_path="config", config_name="train")
def main(cfg):
    _log.info("PyTouch training initialized with the following configuration...")
    _log.info(OmegaConf.to_yaml(cfg))

    _log.info(f"Dataset parameters: {OmegaConf.to_yaml(cfg.data)}")
    _log.info(f"Model parameters: {OmegaConf.to_yaml(cfg.model)}")
    _log.info(f"Training paramters: {OmegaConf.to_yaml(cfg.training)}")
    _log.info(f"Optimizer paramters: {OmegaConf.to_yaml(cfg.optimizer)}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # x = DigitFolder
    # print(x._get_classes(None, cfg.data.path))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    pl.seed_everything(cfg.training.seed)
    # # Instantiate objects from config file
    task_model = instantiate(cfg.model, cfg)
    task_data_module = instantiate(cfg.data, cfg)
    checkpoint_filename = cfg.experiment + "-{epoch}_{val_loss:.3f}_{val_acc:.3f}"

    _log.info(
        f"Creating model checkpoint monitoring {cfg.checkpoints.monitor}, mode: {cfg.checkpoints.mode}"
    )
    _log.info(f"Saving top {cfg.checkpoints.save_top_k} checkpoints!")

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoints.path,
        filename=checkpoint_filename,
        save_top_k=cfg.checkpoints.save_top_k,
        verbose=cfg.general.verbose,
        monitor=cfg.checkpoints.monitor,
        mode=cfg.checkpoints.mode,
        save_weights_only=cfg.checkpoints.save_weights_only,
    )

    logger = TensorBoardLogger(cfg.general.tb_log_path, name=cfg.experiment)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.training.n_epochs,
        callbacks=[checkpoint_callback],
        gpus=1,
        default_root_dir=".",
    )
    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(5, export_path="./")
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(task_model, task_data_module)
    # trainer.fit(task_model, task_data_module)

    _log.info(
        f"Best Checkpoint: {checkpoint_callback.best_model_score} -- {checkpoint_callback.best_model_path}"
    )
    _log.info(f"Training completed for experiment: {cfg.experiment}")

    if cfg.onnx_export:
        _log.info("Exporting to ONNX")
        best_model = task_model.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        input_sample = torch.randn(1, 3, 64, 64)
        onnx_filename = os.path.basename(checkpoint_callback.best_model_path)
        best_model.to_onnx(onnx_filename, input_sample, export_params=True)


if __name__ == "__main__":
    sys.exit(main())
