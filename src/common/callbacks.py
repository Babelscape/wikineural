import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class LinkBestModelCheckpoint(ModelCheckpoint):
    CHECKPOINT_NAME_BEST = "best.ckpt"

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)

        save_dir = Path(self.dirpath)
        save_dir_fd = os.open(save_dir, os.O_RDONLY)
        if self.best_model_path != "":
            orig_best = Path(self.best_model_path)
            save_dir = orig_best.parent
            (save_dir / self.CHECKPOINT_NAME_BEST).unlink(missing_ok=True)
            os.symlink(orig_best.name, self.CHECKPOINT_NAME_BEST, dir_fd=save_dir_fd)

        if self.last_model_path != "":
            orig_last = Path(self.last_model_path)
            (save_dir / ModelCheckpoint.CHECKPOINT_NAME_LAST).unlink(missing_ok=True)
            os.symlink(
                orig_last.name, ModelCheckpoint.CHECKPOINT_NAME_LAST, dir_fd=save_dir_fd
            )

        os.close(save_dir_fd)


class PaddingMonitor(pl.Callback):
    def __init__(self):
        self._tokens = 0
        self._total = 0

    @property
    def percentage(self):
        return 1 - self._tokens / self._total

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        n_tokens = (
            (batch["labels"] != pl_module.hparams.label_encoder.padding_index)
            .sum()
            .item()
        )
        self._tokens += n_tokens
        self._total += batch["labels"].numel()
        pl_module.log(
            "padding", self.percentage, prog_bar=True, on_step=True, on_epoch=False
        )
