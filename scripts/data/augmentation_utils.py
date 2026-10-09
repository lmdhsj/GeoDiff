#!/usr/bin/env python3
"""Shared split, geometry, and manifest utilities for absolute CST augmentation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aerosandbox as asb
import numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates


DESCRIPTOR_NAMES = (
    "max_thickness",
    "max_thickness_position",
    "max_camber",
    "max_camber_position",
)



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dat(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float).reshape(-1)
    if values.size != 20:
        raise ValueError(f"{path.name}: expected 20 values, found {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name}: contains a non-finite value")
    return values


def group_by_content(files: list[Path]) -> list[list[Path]]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_hash[sha256(path)].append(path)
    groups = [sorted(group, key=lambda item: item.name.casefold()) for group in by_hash.values()]
    groups.sort(key=lambda group: group[0].name.casefold())
    return groups


def choose_exact_file_count(
    groups: list[list[Path]], target_file_count: int, seed: int
) -> set[int]:
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for group_index in order:
        size = len(groups[group_index])
        for subtotal in sorted(tuple(reachable), reverse=True):
            new_total = subtotal + size
            if new_total <= target_file_count and new_total not in reachable:
                reachable[new_total] = reachable[subtotal] + (group_index,)
        if target_file_count in reachable:
            break
    if target_file_count not in reachable:
        raise RuntimeError(
            f"Could not select exactly {target_file_count} files while keeping "
            "identical-content groups together."
        )
    return set(reachable[target_file_count])


def make_train_validation_split(
    development_files: list[Path], validation_ratio: float, seed: int
) -> tuple[list[Path], list[Path], int]:
    groups = group_by_content(development_files)
    target_validation = round(len(development_files) * validation_ratio)
    validation_group_indices = choose_exact_file_count(groups, target_validation, seed)
    train_files: list[Path] = []
    validation_files: list[Path] = []
    for group_index, group in enumerate(groups):
        if group_index in validation_group_indices:
            validation_files.extend(group)
        else:
            train_files.extend(group)
    train_files.sort(key=lambda item: item.name.casefold())
    validation_files.sort(key=lambda item: item.name.casefold())
    duplicate_group_count = sum(len(group) > 1 for group in groups)
    return train_files, validation_files, duplicate_group_count


def source_rng(seed: int, filename: str) -> tuple[np.random.Generator, int]:
    material = f"{seed}:{filename}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(derived_seed), derived_seed


def cst_param_to_coords(cst_param: np.ndarray, n_points_per_side: int) -> np.ndarray:
    coords = get_kulfan_coordinates(
        upper_weights=cst_param[:7],
        lower_weights=cst_param[7:14],
        TE_thickness=max(0.0, float(cst_param[14])),
        leading_edge_weight=float(cst_param[15]),
        N1=0.5,
        N2=1.0,
        n_points_per_side=n_points_per_side,
    )
    return np.asarray(coords, dtype=float)


def split_surfaces(coords: np.ndarray, tolerance: float):
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 5:
        raise ValueError("coordinate array has an invalid shape")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinate array contains a non-finite value")

    x_full = coords[:, 0]
    y_full = coords[:, 1]
    i_le = int(np.argmin(x_full))
    if i_le == 0 or i_le == len(x_full) - 1:
        raise ValueError("leading edge does not split upper and lower surfaces")

    x_upper, y_upper = x_full[: i_le + 1], y_full[: i_le + 1]
    x_lower, y_lower = x_full[i_le:], y_full[i_le:]
    if x_upper[0] > x_upper[-1]:
        x_upper, y_upper = x_upper[::-1], y_upper[::-1]
    if x_lower[0] > x_lower[-1]:
        x_lower, y_lower = x_lower[::-1], y_lower[::-1]

    if np.any(np.diff(x_upper) < -tolerance):
        raise ValueError("upper-surface x coordinates are not monotonic")
    if np.any(np.diff(x_lower) < -tolerance):
        raise ValueError("lower-surface x coordinates are not monotonic")
    return x_upper, y_upper, x_lower, y_lower


def evaluate_geometry(
    cst_param: np.ndarray,
    n_points_per_side: int,
    descriptor_method: str,
    uniform_points: int,
    tolerance: float,
) -> tuple[bool, np.ndarray | None, str]:
    try:
        coords = cst_param_to_coords(cst_param, n_points_per_side)
        x_upper, y_upper, x_lower, y_lower = split_surfaces(coords, tolerance)

        # Use the Kulfan upper-surface x grid as the common grid for the
        # validity test. Nonnegative vertical thickness prevents an upper/lower
        # surface crossing for this single-valued CST representation.
        y_lower_on_upper = np.interp(x_upper, x_lower, y_lower)
        node_thickness = y_upper - y_lower_on_upper
        if not np.all(np.isfinite(node_thickness)):
            return False, None, "non-finite thickness"
        if float(np.min(node_thickness)) < -tolerance:
            return False, None, "negative thickness / surface crossing"
        if float(np.max(node_thickness)) <= tolerance:
            return False, None, "non-positive maximum thickness"

        if descriptor_method == "kulfan_nodes":
            x_eval = x_upper
            y_upper_eval = y_upper
            y_lower_eval = y_lower_on_upper
        elif descriptor_method == "uniform_200":
            x_eval = np.linspace(0.0, 1.0, uniform_points)
            y_upper_eval = np.interp(x_eval, x_upper, y_upper)
            y_lower_eval = np.interp(x_eval, x_lower, y_lower)
        else:
            raise ValueError(f"unknown descriptor method: {descriptor_method}")

        thickness = y_upper_eval - y_lower_eval
        camber = 0.5 * (y_upper_eval + y_lower_eval)
        i_thickness = int(np.argmax(thickness))
        i_camber = int(np.argmax(camber))
        descriptors = np.array(
            [
                thickness[i_thickness],
                x_eval[i_thickness],
                camber[i_camber],
                x_eval[i_camber],
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(descriptors)):
            return False, None, "non-finite descriptor"
        return True, descriptors, "valid"
    except Exception as exc:
        return False, None, f"geometry error: {exc}"


def save_dat(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values.reshape(1, -1), fmt="%.8f")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_original(source: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source.name
    if destination.exists():
        raise FileExistsError(f"Output filename collision: {destination.name}")
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Hash mismatch after copying {source.name}")
    return destination


def verify_disjoint_hashes(split_files: dict[str, list[Path]]) -> None:
    split_hashes = {
        split: {sha256(path) for path in files} for split, files in split_files.items()
    }
    names = list(split_hashes)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            overlap = split_hashes[first] & split_hashes[second]
            if overlap:
                raise RuntimeError(
                    f"{len(overlap)} identical-content groups cross {first}/{second}"
                )

