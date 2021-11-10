import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import hydra
import omegaconf
from omegaconf import ValueNode
from torch.utils.data import Dataset
from tqdm import tqdm

from src.common.utils import PROJECT_ROOT

UNK_TOKEN: str = ">>UNK<<"


class MyDataset(Dataset):
    def __init__(
        self,
        name: ValueNode,
        source: str,
        language: str,
        root_path: str,
        split: str,
        idx_in: int,
        idx_out: int,
        sentence_limit: Optional[int] = None,
        labels_map: Optional[Dict[str, str]] = None,
        default_label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.source = source
        self.language = language
        self.root_path = Path(root_path)
        self.split: str = split

        self.labels_map = labels_map
        self.default_label = default_label

        self.idx_in: int = idx_in
        self.idx_out: int = idx_out
        self.sentence_limit: Optional[int] = sentence_limit

        self.samples: List[Dict] = self._read_data()

    def _read_data(self) -> List[Dict]:
        src_path: Path = (
            self.root_path / self.source / self.language / f"{self.split}.conllu"
        )

        sentences: List = []
        skipped_sentences = 0
        with open(src_path, "r", encoding="utf-8") as fr:
            text = ""
            tokens = []
            labels = []

            for i, line in tqdm(
                enumerate(fr), desc=f"[{self.split}] Loading {src_path}"
            ):
                # if i > 100_000:
                #     break
                line = line.strip()

                if line.startswith("# text = "):
                    text = line.strip()[len("# text = ") :].strip()

                if line.startswith("#"):
                    continue

                if "-DOCSTART-" in line:
                    continue

                if line:
                    fields = line.split("\t")
                    token = fields[self.idx_in]
                    label = fields[self.idx_out]

                    if "-" in fields[0]:
                        start, end = fields[0].split("-")
                        num_iter = int(end) - int(start) + 1
                        label = "_".join(
                            fr.readline().split("\t")[self.idx_out]
                            for _ in range(num_iter)
                        )

                    if (
                        re.search("\w", token) is not None
                        or re.search("[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~£−€¿]+", token)
                        is not None
                    ):
                        tokens.append(token)
                    else:
                        tokens.append(UNK_TOKEN)

                    if self.labels_map:
                        labels.append(self.labels_map.get(label, label))
                        # assert self.default_label is not None
                        # labels.append(
                        #     self.labels_map.get(label)
                        #     if label in self.labels_map
                        #     else self.default_label
                        # )
                    # elif self.allowed_labels:
                    #     assert self.default_label is not None
                    #     labels.append(
                    #         label
                    #         if label in self.allowed_labels
                    #         else self.default_label
                    #     )
                    else:
                        labels.append(label)
                else:
                    if len(tokens) > 0:
                        text = " ".join(tokens)
                        if len(text) > 500:
                            skipped_sentences += 1
                        else:
                            sentences.append(
                                dict(
                                    sid=f"{self.source}@{self.language}:{self.split}.{len(sentences)}",
                                    split_type=self.split,
                                    text=text
                                    if len(text.strip()) == 0
                                    else text,
                                    tokens=tokens,
                                    labels=labels,
                                )
                            )
                    if (
                        self.sentence_limit is not None
                        and len(sentences) >= self.sentence_limit
                    ):
                        break
                    text = ""
                    tokens = []
                    labels = []
        # print(f"[{self.split}] skipped sentences: {skipped_sentences}")

        return sentences

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, Any]:
        return self.samples[index]

    def __repr__(self) -> str:
        return f"NERDataset(name={self.name}, split={self.split}, source={self.source}, lang={self.language}, sentences={len(self.samples)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )
    for x in dataset:
        print(x)


if __name__ == "__main__":
    main()
