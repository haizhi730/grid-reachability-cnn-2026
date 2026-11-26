"""Train on one- and two-wall maps and evaluate fixed, disjoint holdouts."""

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from grid import (GAP_P, GOAL, SIDE, SPLIT_PLAN, START, Example, bfs_path,
                  build_splits, column_clearance)


SEEDS = (17, 29, 43)
INPUTS = ("raw", "inflated")
EPOCHS = 14
BATCH_SIZE = 128
LEARNING_RATE = 0.001
THRESHOLD = 0.5
TEST_SPLITS = ("id", "ood")
BASELINES = ("point_bfs", "column_clearance", "footprint_bfs")
SCORES = ("accuracy", "precision", "recall", "f1", "fpr")


class ReachabilityCNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, 1),
        )


def tensors(examples: list[Example], variant: str) -> tuple[torch.Tensor, torch.Tensor]:
    grids = np.stack([getattr(example, variant) for example in examples])
    labels = np.array([example.y for example in examples], dtype=np.float32)
    return torch.from_numpy(grids[:, None]).float(), torch.from_numpy(labels)


def validate(model, x, y, loss_fn) -> float:
    model.eval()
    with torch.inference_mode():
        total = 0.0
        for xb, yb in zip(x.split(BATCH_SIZE), y.split(BATCH_SIZE)):
            total += float(loss_fn(model(xb).squeeze(1), yb)) * len(yb)
    return total / len(y)


def train(variant, seed, x_train, y_train, x_val, y_val, checkpoint_dir):
    torch.manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    model = ReachabilityCNN()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_epoch, best_loss = 0, float("inf")
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(y_train), generator=shuffle)
        total = 0.0
        for indices in order.split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[indices]).squeeze(1)
            loss = loss_fn(logits, y_train[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        average_loss = total / len(y_train)
        val_loss = validate(model, x_val, y_val, loss_fn)
        if val_loss < best_loss:
            best_epoch, best_loss = epoch, val_loss
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"variant": variant, "seed": seed, "epoch": epoch,
                        "train_bce": average_loss, "validation_bce": val_loss})
        print(f"{variant:8s} seed={seed} epoch={epoch:02d} "
              f"train_bce={average_loss:.5f} val_bce={val_loss:.5f}", flush=True)
    training_seconds = time.perf_counter() - start_time
    metadata = {"variant": variant, "seed": seed, "best_epoch": best_epoch,
                "best_validation_bce": best_loss, "training_seconds": training_seconds}
    torch.save({"state": state, "run": metadata}, checkpoint_dir / f"{variant}_{seed}.pt")
    return history


def predict(model, x) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(batch).squeeze(1).sigmoid() for batch in x.split(BATCH_SIZE)]).numpy()


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    tn = int((~y & ~pred).sum())
    fn = int((y & ~pred).sum())
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "fpr": fp / (fp + tn),
    }


def evaluate(splits, checkpoint_dir):
    overall = {split: {} for split in TEST_SPLITS}
    by_walls = {}
    rows = []
    for split in TEST_SPLITS:
        examples = splits[split]
        labels = np.array([example.y for example in examples], dtype=bool)
        walls = np.array([len(example.walls) for example in examples])
        methods = {
            "point_bfs": np.array([bfs_path(example.raw) is not None for example in examples]),
            "column_clearance": np.array([column_clearance(example.inflated) for example in examples]),
            # This reruns the label-generating rule and only checks pipeline consistency.
            "footprint_bfs": np.array([bfs_path(example.inflated) is not None for example in examples]),
        }
        assert np.array_equal(methods["footprint_bfs"], labels)
        assert not np.any(labels & ~methods["point_bfs"])
        assert not np.any(labels & ~methods["column_clearance"])
        probabilities = {}
        for variant in INPUTS:
            x, _ = tensors(examples, variant)
            for seed in SEEDS:
                key = f"{variant}_{seed}"
                checkpoint = torch.load(checkpoint_dir / f"{key}.pt", map_location="cpu", weights_only=True)
                if (checkpoint["run"]["variant"], checkpoint["run"]["seed"]) != (variant, seed):
                    raise ValueError(f"Checkpoint metadata does not match {key}")
                model = ReachabilityCNN()
                model.load_state_dict(checkpoint["state"])
                probabilities[key] = predict(model, x)
                methods[key] = probabilities[key] >= THRESHOLD

        overall[split] = {key: score(labels, pred) for key, pred in methods.items()}
        for wall_count in sorted(set(walls)):
            match = walls == wall_count
            by_walls[f"{split}_{wall_count}"] = {
                key: score(labels[match], pred[match]) for key, pred in methods.items()
            }

        for i, example in enumerate(examples):
            row = {"split": split, "map_id": example.map_id, "walls": len(example.walls), "y": int(example.y)}
            row.update({name: int(methods[name][i]) for name in BASELINES})
            for name, values in probabilities.items():
                row[f"{name}_p"] = float(values[i])
                row[f"{name}_pred"] = int(methods[name][i])
            rows.append(row)
    return overall, by_walls, rows


def summarize_runs(scores):
    summary = {}
    for group, methods in scores.items():
        summary[group] = {}
        for variant in INPUTS:
            runs = [methods[f"{variant}_{seed}"] for seed in SEEDS]
            summary[group][variant] = {
                metric: {"mean": statistics.mean(run[metric] for run in runs),
                         "sample_sd": statistics.stdev(run[metric] for run in runs)}
                for metric in SCORES
            }
    return summary


def describe_splits(splits, skipped):
    return {
        split: {"n": len(examples), "positive": sum(example.y for example in examples),
                "negative": sum(not example.y for example in examples), "skipped_mirrors": skipped[split],
                "by_walls": {wall_count: {
                    "n": sum(len(example.walls) == wall_count for example in examples),
                    "positive": sum(len(example.walls) == wall_count and example.y for example in examples),
                } for wall_count in SPLIT_PLAN[split][1]}}
        for split, examples in splits.items()
    }


def write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_maps(out, splits):
    rows = []
    arrays = {}
    for split, examples in splits.items():
        arrays[split] = np.stack([example.raw for example in examples])
        for example in examples:
            rows.append({"split": split, "map_id": example.map_id, "walls": len(example.walls),
                         "wall_columns": " ".join(map(str, example.walls)), "y": int(example.y),
                         "sha256": example.digest, "mirror_group_sha256": example.group_digest})
    write_csv(out / "map_manifest.csv", rows)
    np.savez_compressed(out / "maps.npz", **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="a new or empty results directory")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--evaluate-from", type=Path, help="skip training and use this checkpoint directory")
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("--out must be a new or empty directory")
    if args.evaluate_from and not all((args.evaluate_from / f"{variant}_{seed}.pt").is_file()
                                      for variant in INPUTS for seed in SEEDS):
        parser.error("--evaluate-from must contain all six checkpoints")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    splits, skipped = build_splits()
    counts = describe_splits(splits, skipped)
    save_maps(args.out, splits)
    print(json.dumps(counts), flush=True)

    checkpoint_dir = args.evaluate_from or (args.out / "checkpoints")
    if not args.evaluate_from:
        checkpoint_dir.mkdir()
        history = []
        for variant in INPUTS:
            x_train, y_train = tensors(splits["train"], variant)
            x_val, y_val = tensors(splits["validation"], variant)
            for seed in SEEDS:
                history.extend(train(variant, seed, x_train, y_train, x_val, y_val, checkpoint_dir))
        write_csv(args.out / "training_history.csv", history)

    full, by_walls, rows = evaluate(splits, checkpoint_dir)
    write_csv(args.out / "test_predictions.csv", rows)
    runs = [torch.load(checkpoint_dir / f"{variant}_{seed}.pt", map_location="cpu", weights_only=True)["run"]
            for variant in INPUTS for seed in SEEDS]
    report = {
        "settings": {"side": SIDE, "start": START, "goal": GOAL, "footprint": [3, 3],
                     "split_plan": SPLIT_PLAN, "gap_width_probabilities": GAP_P,
                     "seeds": SEEDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                     "learning_rate": LEARNING_RATE, "probability_threshold": THRESHOLD,
                     "parameter_count": sum(w.numel() for w in ReachabilityCNN().parameters())},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "system": platform.system(), "os_version": platform.mac_ver()[0] or platform.release(),
                        "machine": platform.machine(), "processor": platform.processor(),
                        "device": "cpu", "threads": args.threads},
        "data": counts, "runs": runs,
        "full": full, "full_summary": summarize_runs(full),
        "by_walls": by_walls, "by_walls_summary": summarize_runs(by_walls),
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
