"""Audit a result directory using a second reachability implementation."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from grid import GOAL, SIDE, START, map_hashes


def independent_collision(raw):
    padded = np.pad(raw, ((0, 0), (1, 1), (1, 1)), constant_values=True)
    windows = sliding_window_view(padded, (3, 3), axis=(1, 2))
    return windows.any(axis=(-2, -1))


def reachable_area(blocked):
    free = ~blocked
    reached = np.zeros_like(free)
    reached[:, START[0], START[1]] = free[:, START[0], START[1]]
    while True:
        expanded = reached.copy()
        expanded[:, 1:, :] |= reached[:, :-1, :]
        expanded[:, :-1, :] |= reached[:, 1:, :]
        expanded[:, :, 1:] |= reached[:, :, :-1]
        expanded[:, :, :-1] |= reached[:, :, 1:]
        expanded &= free
        if np.array_equal(expanded, reached):
            return expanded
        reached = expanded


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def audit_scores(rows, stats, threshold):
    y = np.array([int(row["y"]) for row in rows])
    for method, recorded in stats.items():
        key = f"{method}_pred" if f"{method}_p" in rows[0] else method
        pred = np.array([int(row[key]) for row in rows])
        if key != method:
            assert all(int(float(row[f"{method}_p"]) >= threshold) == int(row[key]) for row in rows)
        tn, fp, fn, tp = np.bincount(2 * y + pred, minlength=4)
        assert [tn, fp, fn, tp] == [recorded[k] for k in ("tn", "fp", "fn", "tp")]
        expected = {"accuracy": (tp + tn) / len(y), "recall": tp / (tp + fn),
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "f1": 2 * tp / (2 * tp + fp + fn), "fpr": fp / (fp + tn)}
        assert all(math.isclose(value, recorded[k], abs_tol=1e-12) for k, value in expected.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    out = parser.parse_args().results
    manifest = read_csv(out / "map_manifest.csv")
    predictions = read_csv(out / "test_predictions.csv")
    report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    threshold = report["settings"]["probability_threshold"]
    assert len({row["mirror_group_sha256"] for row in manifest}) == len(manifest)
    assert len({row["map_id"] for row in manifest}) == len(manifest)
    assert len(predictions) == sum(row["split"] in report["full"] for row in manifest)
    with np.load(out / "maps.npz") as arrays:
        assert set(arrays.files) == set(report["data"])
        for split in arrays.files:
            raw = arrays[split]
            data = report["data"][split]
            rows = [row for row in manifest if row["split"] == split]
            assert raw.shape == (len(rows), SIDE, SIDE) and raw.dtype == np.bool_
            assert len(rows) == data["n"]
            for grid, row in zip(raw, rows):
                assert map_hashes(grid) == (row["sha256"], row["mirror_group_sha256"])
            blocked = independent_collision(raw)
            label = reachable_area(blocked)[:, GOAL[0], GOAL[1]]
            assert np.array_equal(label, [int(row["y"]) for row in rows])
            assert int(label.sum()) == data["positive"]
            if split in report["full"]:
                tests = [row for row in predictions if row["split"] == split]
                assert [(r["map_id"], r["y"], r["walls"]) for r in tests] == [
                    (r["map_id"], r["y"], r["walls"]) for r in rows]
                point = reachable_area(raw)[:, GOAL[0], GOAL[1]]
                column = (~blocked[:, :, START[1]:GOAL[1] + 1]).any(axis=1).all(axis=1)
                for key, truth in [("point_bfs", point), ("column_clearance", column), ("footprint_bfs", label)]:
                    assert np.array_equal(truth, [int(row[key]) for row in tests])
                audit_scores(tests, report["full"][split], threshold)
                for wall_count in data["by_walls"]:
                    group = [row for row in tests if row["walls"] == wall_count]
                    audit_scores(group, report["by_walls"][f"{split}_{wall_count}"], threshold)
            print(f"{split}: verified {len(rows)} maps, {int(label.sum())} reachable")
    print("Passed: distinct maps, independent labels, baseline predictions and reported single-run metrics.")


if __name__ == "__main__":
    main()
