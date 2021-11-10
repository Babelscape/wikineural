from argparse import ArgumentParser
from pathlib import Path
from typing import List

import wandb
from datetime import datetime

from tqdm.auto import tqdm
from wandb.apis.public import Run

from common.utils import PROJECT_ROOT

WANDB_DIR: Path = PROJECT_ROOT / "wandb"


def restore_artifacts(run: Run, run_dir: Path, best_only: bool = True):
    artifacts = run.logged_artifacts()
    if len(artifacts) == 0:
        files = [
            file
            for file in run.files()
            if "checkpoint" in file.name and ('best.ckpt' in file.name or not best_only)
        ]

        assert len(files) > 0, (
            f"There is no file to download from this run! Check on WandB: {run.url}"
        )

        for file in tqdm(files, desc=f"Downloading files for {run.name} ({run.id})"):
            file.download(root=run_dir)
            (run_dir / 'checkpoints').mkdir(exist_ok=True)
            (run_dir / file.name).rename(run_dir / 'checkpoints' / 'model.ckpt')
    else:

        checkpoint = run.logged_artifacts()[0].get_path('model.ckpt')
        checkpoint.download(root=run_dir / 'checkpoints')

    return run_dir.parent


def restore_run(run: Run, best_only: bool = True):
    run_id = run.id

    matching_runs: List[Path] = [
        item
        for item in WANDB_DIR.iterdir()
        if item.is_dir() and item.name.endswith(run_id)
    ]

    assert len(matching_runs) <= 1, (
        f"More than one run matching unique id {run_id}! Are you sure about that?"
    )

    if len(matching_runs) == 1:
        return restore_artifacts(run, matching_runs[0] / 'files', best_only)

    created_at: datetime = datetime.strptime(
        run.created_at, "%Y-%m-%dT%H:%M:%S"
    )

    timestamp: str = created_at.strftime("%Y%m%d_%H%M%S")

    run_dir: Path = WANDB_DIR / f"restored-{timestamp}-{run.id}" / "files"

    return restore_artifacts(run, run_dir, best_only)


def get_run(run_id: str, entity: str, project: str):
    return wandb.Api().run(f"{entity}/{project}/{run_id}")


def get_runs_matching(pattern: str,
                      *,
                      entity: str,
                      project: str,
                      exact=False,
                      tags=None):
    api = wandb.Api()

    filters = dict(
        displayName=pattern if exact else {'$regex': pattern}
    )

    # if tags is not None:
    #     pass  # TODO

    return list(api.runs(f"{entity}/{project}", filters=filters))


def parse_args():
    parser = ArgumentParser()

    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--id')
    grp.add_argument('--name')

    return parser.parse_args()


def main(args):
    assert args.id or args.name

    if args.id:
        runs = [get_run(args.id)]
    else:
        runs = get_runs_matching(args.name)

    for run in runs:
        print(restore_run(run))


if __name__ == '__main__':
    main(parse_args())
