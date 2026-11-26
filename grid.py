"""Procedural occupancy grids and exact four-connected reachability."""

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


SIDE = 25
START = (12, 2)  # zero-based (row, column)
GOAL = (12, 22)
GAP_P = (0.06, 0.10, 0.14, 0.19, 0.19, 0.18, 0.14)
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
WALL_OPTIONS = {
    n: [cols for cols in combinations(range(5, 20), n)
        if all(b - a >= 4 for a, b in zip(cols, cols[1:]))]
    for n in (1, 2, 3)
}
# Each entry lists the map seed and counts of layouts by wall count.
SPLIT_PLAN = {
    "train": (202601, {1: 2500, 2: 2500}),
    "validation": (202602, {1: 400, 2: 400}),
    "id": (202603, {1: 800, 2: 800}),
    "ood": (202604, {3: 1600}),
}
PREFIX = {"train": "T", "validation": "V", "id": "I", "ood": "O"}

Grid = NDArray[np.bool_]
Cell = tuple[int, int]


@dataclass(frozen=True)
class Example:
    map_id: str
    walls: tuple[int, ...]
    raw: Grid
    inflated: Grid
    digest: str
    group_digest: str
    y: bool


def generate_grid(rng: np.random.Generator, wall_count: int) -> tuple[Grid, tuple[int, ...]]:
    """Draw walls with openings, overlay clutter, then clear the endpoints."""
    raw = np.zeros((SIDE, SIDE), dtype=bool)
    raw[[0, -1], :] = True
    raw[:, [0, -1]] = True
    options = WALL_OPTIONS[wall_count]
    walls = options[rng.integers(len(options))]
    for col in walls:
        raw[:, col] = True
        gap_count = 2 if rng.random() < 0.30 else 1
        for _ in range(gap_count):
            width = int(rng.choice(7, p=GAP_P))
            if width:
                row = int(rng.integers(1, SIDE - width))
                raw[row:row + width, col] = False

    for _ in range(rng.integers(3, 9)):
        height, width = rng.integers(1, 5, size=2)
        row = int(rng.integers(1, SIDE - height))
        col = int(rng.integers(1, SIDE - width))
        raw[row:row + height, col:col + width] = True

    for row, col in (START, GOAL):
        raw[row - 1:row + 2, col - 1:col + 2] = False
    return raw, walls


def inflate(raw: Grid) -> Grid:
    """Mark a center blocked if any cell in its square 3-by-3 footprint is blocked."""
    padded = np.pad(raw, 1, constant_values=True)
    height, width = raw.shape
    blocked = np.zeros_like(raw)
    for dr in range(3):
        for dc in range(3):
            blocked |= padded[dr:dr + height, dc:dc + width]
    return blocked


def bfs_path(blocked: Grid, start: Cell = START, goal: Cell = GOAL) -> list[Cell] | None:
    """Return a shortest four-neighbor path, or None if the goal is unreachable."""
    if blocked[start] or blocked[goal]:
        return None
    height, width = blocked.shape
    parents: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        cell = (row, col)
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parents[cell]
            return path[::-1]
        for dr, dc in NEIGHBORS:
            r, c = row + dr, col + dc
            if 0 <= r < height and 0 <= c < width and not blocked[r, c] and (r, c) not in parents:
                parents[r, c] = cell
                queue.append((r, c))
    return None


def column_clearance(blocked: Grid) -> bool:
    """A necessary condition: every column between the endpoints has a free center."""
    return bool((~blocked[:, START[1]:GOAL[1] + 1]).any(axis=0).all())


def map_hashes(raw: Grid) -> tuple[str, str]:
    raw_bytes = np.packbits(raw).tobytes()
    mirrors = (raw_bytes, np.packbits(raw[::-1, :]).tobytes(),
               np.packbits(raw[:, ::-1]).tobytes(), np.packbits(raw[::-1, ::-1]).tobytes())
    return sha256(raw_bytes).hexdigest(), sha256(min(mirrors)).hexdigest()


def build_splits() -> tuple[dict[str, list[Example]], dict[str, int]]:
    splits = {}
    skipped = {}
    seen = set()
    for split, (seed, counts) in SPLIT_PLAN.items():
        rng = np.random.default_rng(seed)
        examples = []
        skipped[split] = 0
        for wall_count, count in counts.items():
            accepted = 0
            while accepted < count:
                raw, walls = generate_grid(rng, wall_count)
                digest, group = map_hashes(raw)
                if group in seen:
                    skipped[split] += 1
                    continue
                seen.add(group)
                blocked = inflate(raw)
                assert not blocked[START] and not blocked[GOAL]
                label = bfs_path(blocked) is not None
                map_id = f"{PREFIX[split]}{len(examples) + 1:04d}"
                examples.append(Example(map_id, walls, raw, blocked, digest, group, label))
                accepted += 1
        splits[split] = examples
    return splits, skipped
