from typing import Any, Dict, Sequence, Tuple, Union, List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Optimizer
from TorchCRF import CRF
from torchmetrics import F1
from transformers import AutoModel, PreTrainedModel

from src.common.utils import PROJECT_ROOT

DEFAULT_K_LAYERS: int = 4


class MyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        self.transformer: PreTrainedModel = AutoModel.from_pretrained(
            self.hparams.transformer_name, output_hidden_states=True, return_dict=True
        )

        if not self.hparams.finetune:
            self.transformer.eval()
            self.transformer.requires_grad_(requires_grad=False)

        self.embedding_size = (
            self.hparams.get("embedding_size", 50) or 0
        )  # for backwards compatibility
        if self.embedding_size > 0:
            self.word_embedding = nn.Embedding(
                self.hparams.token_encoder.vocab_size, self.embedding_size
            )

        self.dropout = nn.Dropout(self.hparams.dropout)

        transformer_config = self.transformer.config.to_dict()
        # TODO: DistilBert doesn't have the "hidden_size" parameter :@
        transformer_encoding_dim = transformer_config[
            "hidden_size" if "hidden_size" in transformer_config else "dim"
        ]
        self.rnn = nn.LSTM(
            input_size=transformer_encoding_dim + self.embedding_size,
            batch_first=True,
            bidirectional=self.hparams.rnn.bidirectional,
            num_layers=self.hparams.rnn.num_layers,
            hidden_size=self.hparams.rnn.hidden_size,
            dropout=self.hparams.rnn.dropout,
        )

        label_vocab_items: int = self.hparams.label_encoder.vocab_size
        self.projection = nn.Linear(
            in_features=self.hparams.rnn.hidden_size
            * (int(self.hparams.rnn.bidirectional) + 1),
            out_features=label_vocab_items,
        )

        self.hparams.setdefault("average_last_k_layers", DEFAULT_K_LAYERS)

        self.cache_representations = True  # TODO: parameterize
        # keeps pooled token-level outputs of the transformer (each tensor has shape (seq_len, hidden_size))
        # tensors in this dictionary must be kept on CPU and detached beforehand
        self._cache: Dict[str, torch.Tensor] = dict()
        self.activation = instantiate(self.hparams.activation)
        self.crf = CRF(label_vocab_items)

        self.val_metrics: nn.ModuleDict = nn.ModuleDict(
            {
                f"f1{f'/{average}' if average != 'none' else ''}": F1(
                    num_classes=label_vocab_items, ignore_index=0, average=average
                )
                for average in ("macro", "micro", "none")
            }
        )

        self.test_metrics: nn.ModuleDict = nn.ModuleDict(
            {
                f"f1/{average if average != 'none' else ''}": F1(
                    num_classes=label_vocab_items, ignore_index=0, average=average
                )
                for average in ("macro", "micro", "none")
            }
        )

    def on_train_epoch_start(self) -> None:
        if not self.hparams.finetune:
            self.transformer.eval()

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        # TODO
        pass

    def compute_loss(self, batch, model_out) -> torch.Tensor:
        logits: torch.Tensor = model_out["logits"]

        target = batch["labels"]

        loss: torch.Tensor = -(
            torch.sum(
                self.crf.forward(
                    logits,
                    target,
                    batch["labels_mask"],
                )
            )
            / len(logits)
        )

        return loss

    def _cached_word_representations(self, batch: Dict[str, Any]):

        sentence_ids = batch["sid"]

        if any(sentence_id not in self._cache for sentence_id in sentence_ids):
            tensors = self._compute_word_representations(batch)
            for sid, tensor in zip(sentence_ids, tensors):
                self._cache[sid] = tensor.detach().cpu()

        return [self._cache[sid].to(self.device, non_blocking=True) for sid in sentence_ids]

    def _compute_word_representations(self, batch) -> List[torch.Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(self.hparams.finetune)
        transformer_out = self.transformer(**batch["encoding"])
        torch.set_grad_enabled(prev)

        encoding: torch.Tensor = torch.stack(
            transformer_out["hidden_states"][-self.hparams.average_last_k_layers :],
            dim=2,
        ).sum(dim=2)

        bpe_weights = torch.sparse.FloatTensor(
            batch["bpe_indices"], batch["bpe_values"], batch["bpe_weigths_size"]
        )

        encoding = torch.bmm(bpe_weights, encoding)

        # Padding and special tokens (CLS & SEP) removal
        encoding = encoding[batch["encoding_mask"]]
        # encoding is now 1-D, we need to split it again
        encoding: List[torch.Tensor] = torch.split(encoding, batch["sentence_lengths"])

        return encoding

    def represent_words(self, batch):
        if self.training and self.cache_representations and not self.hparams.finetune:
            tensors = self._cached_word_representations(batch)
        else:
            tensors = self._compute_word_representations(batch)

        return pad_sequence(tensors, batch_first=True, padding_value=0)

    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        hidden_states = self.represent_words(batch)
        if self.embedding_size > 0:

            word_level_embeddings = self.dropout(self.word_embedding(batch["tokens"]))
            # CLS and SEP have been removed inside self.represent_words
            hidden_states = torch.cat([hidden_states, word_level_embeddings], dim=-1)

        packed_encoding = pack_padded_sequence(
            hidden_states,
            lengths=batch["sentence_lengths"],
            batch_first=True,
            enforce_sorted=False,
        )
        merged_encoding, *_ = self.rnn(packed_encoding)
        merged_encoding, merged_lengths = pad_packed_sequence(
            sequence=merged_encoding,
            batch_first=True,
        )

        logits: torch.Tensor = self.dropout(merged_encoding)
        logits: torch.Tensor = self.projection(logits)
        logits: torch.Tensor = self.activation(logits)

        return dict(logits=logits)

    def step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        compute_loss: bool = True,
        compute_metrics: bool = False,
    ) -> Dict[str, Any]:
        assert split != "train" or not compute_metrics
        model_out: Dict[str, Any] = self(batch)

        result = dict(model_out=model_out)
        if compute_loss:
            result["loss"] = self.compute_loss(batch=batch, model_out=model_out)

        if compute_metrics:
            metrics = self.val_metrics if split == "val" else self.test_metrics
            logits = model_out["logits"]
            preds: List[List[int]] = self.crf.viterbi_decode(
                logits, mask=batch["labels_mask"]
            )
            preds: List[torch.Tensor] = [
                torch.tensor(seq, device=self.device) for seq in preds
            ]
            preds: torch.Tensor = torch.cat(preds, dim=0)

            result["metrics"] = {
                metric_name: metric(preds, batch["labels"][batch["labels_mask"]])
                for metric_name, metric in metrics.items()
            }

        return result

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_out = self.step(batch, batch_idx, split="train", compute_metrics=False)
        self.log_dict(
            {"train_loss": step_out["loss"]},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return step_out["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_out = self.step(batch, batch_idx, split="val", compute_metrics=True)
        self.log_dict(
            {"val_loss": step_out["loss"]},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        f1_simple = {
            f"f1/{label}": score
            for label, score in zip(
                self.hparams.label_encoder.vocab[1:], step_out["metrics"]["f1"][1:]
            )
        }
        del step_out["metrics"]["f1"]

        self.log_dict(step_out["metrics"])
        self.log_dict(f1_simple)

        return step_out["loss"]

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        step_out = self.step(batch, batch_idx, split="test", compute_metrics=True)
        self.log_dict(
            {"test_loss": step_out["loss"]},
        )
        f1_simple = step_out["metrics"]["f1"]
        del step_out["metrics"]["f1"]

        self.log_dict(step_out["metrics"])
        self.log_dict(f1_simple)
        return step_out["loss"]

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]], Dict[str, Any]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return opt
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval="step",
                frequency=1,
            ),
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
