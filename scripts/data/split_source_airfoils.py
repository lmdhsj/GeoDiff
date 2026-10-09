#!/usr/bin/env python3
"""Create a reproducible source-level train/test split for airfoil .dat files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DESCRIPTOR_NAMES = (
    "max_thickness",
    "max_thickness_position",
    "max_camber",
    "max_camber_position",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--expected-count", type=int, default=1500)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(path: Path) -> dict[str, object]:
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) != 20:
        raise ValueError(f"{path.name}: expected 20 values, found {len(tokens)}")
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"{path.name}: contains a non-numeric value") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path.name}: contains a non-finite value")
    return {
        "path": path,
        "filename": path.name,
        "source_id": path.stem,
        "sha256": sha256(path),
        "values": values,
        "descriptors": values[-4:],
    }


def choose_test_groups(groups: list[list[dict[str, object]]], target_files: int, seed: int) -> set[int]:
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)

    # Exact subset-sum selection keeps duplicate-content groups together while
    # preserving the requested number of files in the test subset.
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for group_index in order:
        size = len(groups[group_index])
        for subtotal in sorted(tuple(reachable), reverse=True):
            new_total = subtotal + size
            if new_total <= target_files and new_total not in reachable:
                reachable[new_total] = reachable[subtotal] + (group_index,)
        if target_files in reachable:
            break

    if target_files not in reachable:
        raise RuntimeError(f"Could not create an exact {target_files}-file test split")
    return set(reachable[target_files])


def describe(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, name in enumerate(DESCRIPTOR_NAMES):
        values = [record["descriptors"][index] for record in records]
        rows.append(
            {
                "descriptor": name,
                "count": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    return rows


def copy_and_verify(record: dict[str, object], destination: Path) -> None:
    source = record["path"]
    target = destination / record["filename"]
    shutil.copy2(source, target)
    if sha256(target) != record["sha256"]:
        raise RuntimeError(f"Hash mismatch after copying {record['filename']}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the source directory")
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("test-ratio must be between 0 and 1")

    files = sorted(source_dir.glob("*.dat"), key=lambda path: path.name.casefold())
    if len(files) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} .dat files, found {len(files)}")

    records = [read_record(path) for path in files]
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)

    groups = [sorted(group, key=lambda item: item["filename"].casefold()) for group in by_hash.values()]
    groups.sort(key=lambda group: group[0]["filename"].casefold())
    target_test_files = round(len(records) * args.test_ratio)
    test_group_indices = choose_test_groups(groups, target_test_files, args.seed)

    split_by_filename: dict[str, str] = {}
    group_id_by_filename: dict[str, str] = {}
    for group_index, group in enumerate(groups, start=1):
        split = "test" if (group_index - 1) in test_group_indices else "train"
        group_id = f"content_group_{group_index:04d}"
        for record in group:
            split_by_filename[record["filename"]] = split
            group_id_by_filename[record["filename"]] = group_id

    train_records = [record for record in records if split_by_filename[record["filename"]] == "train"]
    test_records = [record for record in records if split_by_filename[record["filename"]] == "test"]
    if len(train_records) + len(test_records) != len(records):
        raise RuntimeError("Split does not account for every record")
    if len(test_records) != target_test_files:
        raise RuntimeError("Test split size does not match target")

    train_hashes = {record["sha256"] for record in train_records}
    test_hashes = {record["sha256"] for record in test_records}
    if train_hashes & test_hashes:
        raise RuntimeError("Identical file content appears in both train and test splits")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp-", dir=output_dir.parent)).resolve()
    try:
        train_dir = temp_dir / "train_original"
        test_dir = temp_dir / "test_original"
        manifests_dir = temp_dir / "manifests"
        train_dir.mkdir()
        test_dir.mkdir()
        manifests_dir.mkdir()

        for record in train_records:
            copy_and_verify(record, train_dir)
        for record in test_records:
            copy_and_verify(record, test_dir)

        manifest_rows: list[dict[str, object]] = []
        for record in records:
            descriptors = record["descriptors"]
            manifest_rows.append(
                {
                    "filename": record["filename"],
                    "source_id": record["source_id"],
                    "content_group_id": group_id_by_filename[record["filename"]],
                    "split": split_by_filename[record["filename"]],
                    "sha256": record["sha256"],
                    "max_thickness": descriptors[0],
                    "max_thickness_position": descriptors[1],
                    "max_camber": descriptors[2],
                    "max_camber_position": descriptors[3],
                }
            )

        manifest_fields = [
            "filename",
            "source_id",
            "content_group_id",
            "split",
            "sha256",
            *DESCRIPTOR_NAMES,
        ]
        write_csv(manifests_dir / "split_manifest.csv", manifest_fields, manifest_rows)
        write_csv(
            manifests_dir / "train_sources.csv",
            manifest_fields,
            [row for row in manifest_rows if row["split"] == "train"],
        )
        write_csv(
            manifests_dir / "test_sources.csv",
            manifest_fields,
            [row for row in manifest_rows if row["split"] == "test"],
        )

        duplicate_rows: list[dict[str, object]] = []
        for group_index, group in enumerate(groups, start=1):
            if len(group) > 1:
                duplicate_rows.append(
                    {
                        "content_group_id": f"content_group_{group_index:04d}",
                        "split": split_by_filename[group[0]["filename"]],
                        "file_count": len(group),
                        "filenames": " | ".join(record["filename"] for record in group),
                        "sha256": group[0]["sha256"],
                    }
                )
        write_csv(
            manifests_dir / "duplicate_content_groups.csv",
            ["content_group_id", "split", "file_count", "filenames", "sha256"],
            duplicate_rows,
        )

        distribution_rows: list[dict[str, object]] = []
        for split_name, split_records in (("train", train_records), ("test", test_records)):
            for row in describe(split_records):
                distribution_rows.append({"split": split_name, **row})
        write_csv(
            manifests_dir / "descriptor_distribution.csv",
            ["split", "descriptor", "count", "mean", "std", "min", "max"],
            distribution_rows,
        )

        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(source_dir),
            "output_directory": str(output_dir),
            "split_unit": "original .dat source file, with byte-identical files grouped",
            "method": "deterministic random group split",
            "seed": args.seed,
            "requested_test_ratio": args.test_ratio,
            "source_file_count": len(records),
            "unique_content_group_count": len(groups),
            "duplicate_content_group_count": sum(len(group) > 1 for group in groups),
            "train_file_count": len(train_records),
            "test_file_count": len(test_records),
            "actual_test_ratio": len(test_records) / len(records),
            "cross_split_identical_hash_count": len(train_hashes & test_hashes),
        }
        (manifests_dir / "split_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        readme = f"""# Source-level airfoil split for R1 review

This directory is a reproducible 80/20 split of the original 1,500 airfoil
files. The original dataset was not modified.

- Random seed: {args.seed}
- Training files: {len(train_records)}
- Test files: {len(test_records)}
- Unique byte-content groups: {len(groups)}
- Duplicate-content groups kept on one side: {sum(len(group) > 1 for group in groups)}
- Identical hashes shared across train/test: 0

Use `train_original` as the only source for augmentation and model fitting.
Keep `test_original` untouched until final evaluation. Both the autoencoder
and diffusion model must use the same split manifest.

See `manifests/split_metadata.json` and `manifests/split_manifest.csv` for
the exact protocol and membership.
"""
        (temp_dir / "README.md").write_text(readme, encoding="utf-8")

        temp_dir.replace(output_dir)
    except Exception:
        if temp_dir.exists() and temp_dir.parent == output_dir.parent:
            shutil.rmtree(temp_dir)
        raise

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
