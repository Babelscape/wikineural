from argparse import ArgumentParser
from pathlib import Path
from typing import List

import datasets
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from restore import get_run, get_runs_matching, restore_run
from src.pl_data.datamodule import MyDataModule, worker_init_fn
from src.pl_data.dataset import MyDataset
from src.pl_modules.model import MyModel


def parse_args():
    parser = ArgumentParser()

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--checkpoint')
    grp.add_argument('--name')
    grp.add_argument('--id')

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--split", default="test")

    parser.add_argument("--mode", default="print", choices=["eval", "print"])

    parser.add_argument("--suppress-tags", default=None)

    parser.add_argument("--data-path", default="data")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=16, type=int)

    return parser.parse_args()


def get_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint, args.checkpoint

    if args.id:
        run = get_run(args.id)
    else:
        runs = get_runs_matching(args.name, exact=True)
        assert len(runs) == 1, (
            f'specify a unique name for your experiment! {args.name} returns {", ".join(map(lambda r: r.id, runs))}'
        )
        run = runs[0]

    files_path = restore_run(run, best_only=True)

    return str(files_path / 'files' / 'checkpoints' / 'model.ckpt'), run.name


def main(args):

    checkpoint, model_name = get_checkpoint(args)

    model: MyModel = MyModel.load_from_checkpoint(checkpoint_path=checkpoint, map_location=args.device)
    model.to(args.device).freeze()

    label_encoder = model.hparams.label_encoder
    token_encoder = model.hparams.token_encoder

    label_mapping = None

    if args.suppress_tags is not None:
        label_mapping = {}
        to_suppress = args.suppress_tags.split(',')

        for tag in to_suppress:
            label_mapping[f'B-{tag}'] = 'O'
            label_mapping[f'I-{tag}'] = 'O'

    def get_predicted_label(idx):
        predicted_tag = label_encoder.vocab[idx]

        if label_mapping is not None and predicted_tag in label_mapping:
            return 'O'

        return predicted_tag

    tokenizer = AutoTokenizer.from_pretrained(model.transformer.name_or_path)

    split = args.split
    source = args.dataset
    language = args.language
    dataset = MyDataset(
        name=f"{source}@{language}:{split}",
        source=source,
        language=language,
        split=split,
        root_path=args.data_path,
        idx_in=1,
        idx_out=2,
        labels_map=label_mapping,
    )

    def _collate(samples):
        return MyDataModule.collate(
            samples=samples,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            token_encoder=token_encoder,
        )

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=_collate,
    )

    # only populated when mode == 'eval'
    true_by_sentence = []
    pred_by_sentence = []

    for batch in tqdm(loader, desc=f"Predicting {dataset.name} with {model_name}"):
        batch = MyDataModule.transfer_batch_to_device(None, batch, model.device)
        model_out = model(batch)

        preds: List[List[int]] = model.crf.viterbi_decode(
            model_out["logits"], mask=batch["labels_mask"]
        )
        predicted_labels: List[List[str]] = [
            [get_predicted_label(idx) for idx in pred] for pred in preds
        ]
        true_labels: List[List[str]] = [
            [get_predicted_label(idx) for idx in labels] for labels in batch["labels"]
        ]

        # assert predicted_labels == [
        #     [label_encoder.vocab[idx] for idx in labels] for labels in preds
        # ]

        tokens = batch["texts"]

        for pred, true, toks in zip(predicted_labels, true_labels, tokens):
            true = true[: len(toks)]
            assert len(pred) == len(toks)

            if args.mode == 'eval':
                true_by_sentence.append(true)
                pred_by_sentence.append(pred)

            if args.mode == "print":
                for pl, tl, t in zip(pred, true, toks):
                    print(t, "\t", tl, "\t", pl)
                print()

    if args.mode == "eval":
        seqeval = datasets.load_metric('seqeval')
        span_scores = seqeval.compute(predictions=pred_by_sentence, references=true_by_sentence)
        # flatten lists
        full_true = [t for s in true_by_sentence for t in s]
        full_pred = [p for s in pred_by_sentence for p in s]
        p, r, f1, s = precision_recall_fscore_support(full_true, full_pred, average='macro')
        print(f"{f1 * 100:.2f}\t{span_scores['overall_f1'] * 100:.2f}")


if __name__ == "__main__":
    main(parse_args())
