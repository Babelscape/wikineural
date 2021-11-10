import random
from itertools import groupby
from pathlib import Path
from pprint import pprint
from typing import Optional, Sequence

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchnlp.encoders.text import StaticTokenizerEncoder
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.common.utils import PROJECT_ROOT


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


def count_subsequent(sequence):
    lengths = [
        sum(1 for _ in group) for idx, group in groupby(sequence) if idx is not None
    ]
    # CLS and SEP
    lengths.insert(0, 1)
    lengths.append(1)
    return lengths


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        language: str,
        source: str,
        root_path: str,
        transformer_name: str,
        vocab_min_freq: int,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.vocab_min_freq = vocab_min_freq
        self.source: str = source
        self.language: str = language
        self.root_path: Path = Path(root_path)
        assert self.root_path.exists()

        self.transformer_name: str = transformer_name
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            transformer_name
        )

        self.token_encoder = None
        self.label_encoder = None

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def prepare_data(self) -> None:
        # download only
        pass

    # def transfer_batch_to_device(
    #     self, batch: Any, device: Optional[torch.device] = None
    # ) -> Any:
    #     exclude_ids = {"word_lengths", "sid"}
    #     return move_data_to_device(
    #         {k: v for k, v in batch.items() if k not in exclude_ids}, device
    #     )

    @classmethod
    def encoder_from_dataset(
        cls, dataset, key, min_freq=1, specials=("<pad>", "<unk>")
    ):
        return StaticTokenizerEncoder(
            # requires an iterable of whitespace-tokenized strings,
            # so we join the sequence in order for it to be split again...
            (" ".join(item[key]) for item in dataset),
            min_occurrences=min_freq,
            reserved_tokens=list(specials),
        )

    def setup(self, stage: Optional[str] = None):
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train, split="train"
            )
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg, split="val")
                for dataset_cfg in self.datasets.val
            ]

            self.token_encoder = self.encoder_from_dataset(
                self.train_dataset, "tokens", self.vocab_min_freq
            )
            self.label_encoder = self.encoder_from_dataset(
                self.train_dataset, "labels", specials=("<pad>",)
            )

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg, split="test")
                for dataset_cfg in self.datasets.test
            ]

    @staticmethod
    def collate(samples, tokenizer: PreTrainedTokenizer, token_encoder, label_encoder):
        encoding = tokenizer(
            [sample["tokens"] for sample in samples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True,
        )
        sentence_word_lengths = [
            count_subsequent(sample.words) for sample in encoding.encodings
        ]

        texts = [sample["tokens"] for sample in samples]
        tokens = token_encoder.batch_encode(
            " ".join(sample["tokens"]) for sample in samples
        ).tensor

        labels = label_encoder.batch_encode(
            " ".join(sample["labels"]) for sample in samples
        ).tensor

        labels_mask = (labels != label_encoder.padding_index).bool()
        sids = [sample["sid"] for sample in samples]

        offsets = [
            [0] + np.cumsum(word_lengths).tolist()
            for word_lengths in sentence_word_lengths
        ]
        max_offsets_seq = max(len(seq) for seq in offsets)

        *indices, values = list(
            zip(
                *[
                    (i_sentence, i, offset, 1 / (end_offset - start_offset))
                    for i_sentence, sentence_offsets in enumerate(offsets)
                    for i, (start_offset, end_offset) in enumerate(
                        zip(sentence_offsets[:-1], sentence_offsets[1:])
                    )
                    for offset in range(start_offset, end_offset)
                ]
            )
        )
        #
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        weights_size = torch.Size(
            (
                encoding.input_ids.size(0),
                max_offsets_seq - 1,
                encoding.input_ids.size(1),
            )
        )

        zeros = torch.zeros(encoding.input_ids.size(0), 1)
        # Add exclusion of CLS and SEP
        encoding_mask = torch.cat([zeros, labels_mask, zeros], dim=1).bool()

        # We subtract 2 to account for CLS and SEP
        sentence_lengths = [len(s_lengths) - 2 for s_lengths in sentence_word_lengths]

        return dict(
            encoding=encoding,
            labels=labels,
            word_lengths=sentence_word_lengths,
            sentence_lengths=sentence_lengths,
            tokens=tokens,
            labels_mask=labels_mask,
            encoding_mask=encoding_mask,
            texts=texts,
            sid=sids,
            bpe_indices=indices,
            bpe_values=values,
            bpe_weigths_size=weights_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            collate_fn=lambda x: MyDataModule.collate(
                samples=x,
                tokenizer=self.tokenizer,
                label_encoder=self.label_encoder,
                token_encoder=self.token_encoder,
            ),
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                pin_memory=True,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                collate_fn=lambda x: MyDataModule.collate(
                    samples=x,
                    tokenizer=self.tokenizer,
                    label_encoder=self.label_encoder,
                    token_encoder=self.token_encoder,
                ),
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                pin_memory=True,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                collate_fn=lambda x: MyDataModule.collate(
                    samples=x,
                    tokenizer=self.tokenizer,
                    label_encoder=self.label_encoder,
                    token_encoder=self.token_encoder,
                ),
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")
    pprint(next(iter(datamodule.test_dataloader()[0])))


if __name__ == "__main__":
    main()
