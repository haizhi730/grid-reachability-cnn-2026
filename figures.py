"""Render report figures from the saved maps, training log and predictions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from grid import GOAL, START, bfs_path, inflate
from verify_results import reachable_area, read_csv


WHITE = "#fbfaf7"
INK = "#283742"
BLOCKED = "#becbd1"
PATH = "#007572"
REACHED = "#aedfd4"
GOAL_COLOR = "#a64627"
COLORS = ListedColormap([WHITE, BLOCKED, INK, REACHED])


def panel(ax, raw, title, blocked=None, path=None, visited=None, footprint=None):
    data = np.zeros(raw.shape, dtype=int)
    if blocked is not None:
        data[blocked] = 1
    data[raw] = 2
    if visited is not None:
        data[visited] = 3
    ax.imshow(data, cmap=COLORS, vmin=0, vmax=3, interpolation="nearest")
    ax.set_xticks(np.arange(raw.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(raw.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="#64717b", linewidth=.23, alpha=.28)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if path is not None:
        rows, columns = np.asarray(path).T
        ax.plot(columns, rows, color=PATH, lw=1.6, zorder=3, solid_capstyle="round")
    if footprint is not None:
        row, col = footprint
        ax.add_patch(Rectangle((col - 1.5, row - 1.5), 3, 3, edgecolor=GOAL_COLOR,
                               linewidth=1.0, facecolor="none", linestyle="--", zorder=4))
    for (r, c), name, color in [(START, "S", PATH), (GOAL, "G", GOAL_COLOR)]:
        ax.scatter(c, r, s=59, c=color, edgecolors="white", linewidths=.8, zorder=5)
        ax.text(c, r + .03, name, ha="center", va="center", fontsize=6.1,
                fontweight="bold", color="white", zorder=6)
    ax.set_title(title, fontsize=9, color=INK, loc="left", pad=5)


def raw_for(maps, record):
    return maps[record["split"]][int(record["map_id"][1:]) - 1]


def positive_map(maps, predictions, rows_by_id, out):
    for row in predictions:
        if row["split"] == "id" and row["walls"] == "1" and row["y"] == "1":
            raw = raw_for(maps, row)
            blocked = inflate(raw)
            if blocked[START[0], START[1]:GOAL[1] + 1].any():
                break
    else:
        raise ValueError("No positive map with a detour")
    path = bfs_path(blocked)
    wall = int(rows_by_id[row["map_id"]]["wall_columns"])
    crossing = next(cell for cell in path if cell[1] == wall)
    fig, axes = plt.subplots(1, 2, figsize=(5.38, 2.69))
    fig.subplots_adjust(left=.015, right=.985, top=.91, bottom=.145, wspace=.13)
    panel(axes[0], raw, "(a) Original occupancy", path=path, footprint=crossing)
    panel(axes[1], raw, "(b) Center collision grid", blocked=blocked, path=path)
    keys = [Patch(color=INK, label="Obstacle"), Patch(color=BLOCKED, label="Extra blocked"),
            Line2D([], [], lw=1.6, color=PATH, label="Valid path"),
            Patch(edgecolor=GOAL_COLOR, fill=False, linestyle="--", label="3 x 3 footprint")]
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(.5, .016), ncol=4,
               frameon=False, fontsize=7.2, columnspacing=1.45)
    fig.savefig(out / "task_map.png", dpi=300)
    plt.close(fig)
    return {"map_id": row["map_id"], "path_steps": len(path) - 1, "crossing": crossing}


def loss_curves(history, out):
    fig, axes = plt.subplots(1, 2, figsize=(5.38, 2.05), sharey=True)
    fig.subplots_adjust(left=.105, right=.982, top=.84, bottom=.245, wspace=.17)
    for ax, variant, label in zip(axes, ("raw", "inflated"), ("Raw-grid CNN", "Inflated-grid CNN")):
        entries = [row for row in history if row["variant"] == variant]
        seeds = sorted({row["seed"] for row in entries})
        epochs = sorted({int(row["epoch"]) for row in entries})
        for field, name, color, line in [("train_bce", "Train", PATH, "-"),
                                          ("validation_bce", "Validation", GOAL_COLOR, "--")]:
            values = np.array([[float(row[field]) for row in entries if row["seed"] == seed] for seed in seeds])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(epochs, mean, linestyle=line, lw=1.6, color=color, label=name)
            ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=.13, edgecolor="none")
        ax.set_title(label, loc="left", fontsize=9, color=INK, pad=5)
        ax.set_xticks([1, 4, 7, 10, 14])
        ax.set_xlim(.7, 14.3)
        ax.set_ylim(0, .76)
        ax.grid(axis="y", color="#d7dcdf", lw=.5)
        ax.tick_params(length=2, labelsize=7.3, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#7b8790")
    axes[0].set_ylabel("Binary cross-entropy", fontsize=7.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(.5, .001), frameon=False, fontsize=7.6)
    fig.text(.97, .083, "Epoch", ha="right", fontsize=7.6, color=INK)
    fig.savefig(out / "loss_curves.png", dpi=300)
    plt.close(fig)


def failure_maps(maps, predictions, rows_by_id, out):
    narrow = [r for r in predictions if r["split"] == "id" and r["walls"] == "1" and r["y"] == "0"
              and r["point_bfs"] == "1" and r["column_clearance"] == "0"]
    example_a = next((r for r in narrow if r["raw_17_pred"] == "1" and r["inflated_17_pred"] == "0"), narrow[0])
    disconnected = [r for r in predictions if r["split"] == "ood" and r["y"] == "0" and r["column_clearance"] == "1"]
    example_b = next((r for r in disconnected if r["inflated_17_pred"] == "1"), disconnected[0])
    raw_a = raw_for(maps, example_a)
    raw_b = raw_for(maps, example_b)
    inflated_b = inflate(raw_b)
    from_start = reachable_area(inflated_b[None])[0]
    fig, axes = plt.subplots(1, 3, figsize=(5.38, 2.13))
    fig.subplots_adjust(left=.015, right=.985, top=.875, bottom=.12, wspace=.12)
    panel(axes[0], raw_a, f"(a) {example_a['map_id']}  Point path", path=bfs_path(raw_a))
    panel(axes[1], raw_a, f"(b) {example_a['map_id']}  Inflated", blocked=inflate(raw_a))
    panel(axes[2], raw_b, f"(c) {example_b['map_id']}  Inflated", blocked=inflated_b, visited=from_start)
    keys = [Patch(color=INK, label="Obstacle"), Patch(color=BLOCKED, label="Extra blocked"),
            Line2D([], [], lw=1.6, color=PATH, label="Point path"), Patch(color=REACHED, label="Reachable from S")]
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(.5, .002), ncol=4,
               frameon=False, fontsize=7.1, columnspacing=1.55)
    fig.savefig(out / "failure_maps.png", dpi=300)
    plt.close(fig)
    return {"narrow": {**example_a, "wall_columns": rows_by_id[example_a["map_id"]]["wall_columns"]},
            "disconnected": {**example_b, "wall_columns": rows_by_id[example_b["map_id"]]["wall_columns"],
                             "connected_cells": int(from_start.sum())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    predictions = read_csv(args.results / "test_predictions.csv")
    manifest = {row["map_id"]: row for row in read_csv(args.results / "map_manifest.csv")}
    with np.load(args.results / "maps.npz") as archive:
        maps = {key: archive[key] for key in ("id", "ood")}
    cases = {"positive": positive_map(maps, predictions, manifest, out),
             **failure_maps(maps, predictions, manifest, out)}
    if (args.results / "training_history.csv").is_file():
        loss_curves(read_csv(args.results / "training_history.csv"), out)
    (out / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
