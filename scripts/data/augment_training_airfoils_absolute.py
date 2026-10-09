#!/usr/bin/env python3
"""Build a leakage-free dataset with absolute CST-parameter perturbations.

Default protocol
----------------
1. Read the existing source-level 1,200/300 development/test split.
2. Reproduce the 1,080/120 train/validation split using seed 44.
3. Copy every original training source unchanged.
4. For every training source, generate 20 valid variants by adding an
   independent Uniform(-0.05, +0.05) absolute offset to dimensions 1-14.
5. Keep dimensions 15-16 unchanged and recompute dimensions 17-20 from the
   generated Kulfan geometry.
6. Leave validation and test samples unaugmented.

This script imports the verified geometry, hashing, and split utilities from
``augmentation_utils.py`` in the same directory. The input directory is
never modified. Output is created atomically, and an existing output directory
is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aerosandbox as asb
import numpy as np

import augmentation_utils as base


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "dataset/reproduced/source_split_r1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "dataset/reproduced/source_split_r1_aug20_abs005"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a source-level dataset using absolute CST perturbations."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="Fraction of the 1,200 development sources used for validation.",
    )
    parser.add_argument("--augmentations-per-source", type=int, default=20)
    parser.add_argument(
        "--absolute-perturbation",
        type=float,
        default=0.05,
        help=(
            "Maximum absolute offset added independently to each of dimensions "
            "1-14; offsets are sampled uniformly from [-value, +value]."
        ),
    )
    parser.add_argument("--n-points-per-side", type=int, default=100)
    parser.add_argument(
        "--descriptor-method",
        choices=("kulfan_nodes", "uniform_200"),
        default="kulfan_nodes",
    )
    parser.add_argument("--uniform-points", type=int, default=200)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-attempts-per-source", type=int, default=5000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show the plan without creating files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation-ratio must be between 0 and 1")
    if args.augmentations_per_source < 1:
        raise ValueError("augmentations-per-source must be positive")
    if not np.isfinite(args.absolute_perturbation):
        raise ValueError("absolute-perturbation must be finite")
    if args.absolute_perturbation <= 0.0:
        raise ValueError("absolute-perturbation must be positive")
    if args.n_points_per_side < 10:
        raise ValueError("n-points-per-side is unexpectedly small")
    if args.uniform_points < 10:
        raise ValueError("uniform-points is unexpectedly small")
    if args.max_attempts_per_source < args.augmentations_per_source:
        raise ValueError(
            "max-attempts-per-source cannot be smaller than augmentations-per-source"
        )


def finalize_output(temp_root: Path, output_root: Path) -> None:
    """Retry transient directory locks; leave completed data intact on failure."""
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        if output_root.exists():
            raise FileExistsError(
                f"Output already exists and will not be overwritten: {output_root}"
            )
        try:
            # Windows rename refuses to overwrite an existing destination.
            temp_root.rename(output_root)
            return
        except PermissionError:
            if attempt == max_attempts:
                raise
            print(
                f"Output directory rename denied (attempt {attempt}/{max_attempts}); "
                "retrying in 2 seconds...",
                flush=True,
            )
            time.sleep(2.0)


def main() -> None:
    args = parse_args()
    validate_args(args)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    development_dir = input_root / "train_original"
    test_input_dir = input_root / "test_original"

    if not development_dir.is_dir() or not test_input_dir.is_dir():
        raise FileNotFoundError(
            "Expected train_original and test_original under " f"{input_root}"
        )
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_root}"
        )
    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("Output root must not be inside the input root")

    development_files = sorted(
        development_dir.glob("*.dat"), key=lambda item: item.name.casefold()
    )
    test_files = sorted(
        test_input_dir.glob("*.dat"), key=lambda item: item.name.casefold()
    )
    if len(development_files) != 1200 or len(test_files) != 300:
        raise ValueError(
            "Expected 1,200 development files and 300 test files, found "
            f"{len(development_files)} and {len(test_files)}"
        )

    # Parse all sources before creating any output.
    for path in development_files + test_files:
        base.load_dat(path)

    train_sources, validation_sources, duplicate_groups = (
        base.make_train_validation_split(
            development_files, args.validation_ratio, args.seed
        )
    )
    base.verify_disjoint_hashes(
        {
            "train": train_sources,
            "validation": validation_sources,
            "test": test_files,
        }
    )

    planned_train_files = len(train_sources) * (
        1 + args.augmentations_per_source
    )
    plan = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "train_source_count": len(train_sources),
        "validation_source_count": len(validation_sources),
        "test_source_count": len(test_files),
        "augmentations_per_train_source": args.augmentations_per_source,
        "planned_train_file_count": planned_train_files,
        "perturbation_type": "absolute_additive_uniform",
        "absolute_perturbation": args.absolute_perturbation,
        "descriptor_method": args.descriptor_method,
        "development_duplicate_content_groups": duplicate_groups,
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, **plan}, ensure_ascii=False, indent=2))
        return

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f"{output_root.name}.tmp-", dir=output_root.parent)
    ).resolve()

    try:
        train_output_dir = temp_root / "train_combined"
        validation_output_dir = temp_root / "validation_original"
        test_output_dir = temp_root / "test_original"
        manifests_dir = temp_root / "manifests"
        for directory in (
            train_output_dir,
            validation_output_dir,
            test_output_dir,
            manifests_dir,
        ):
            directory.mkdir()

        source_rows: list[dict[str, object]] = []
        train_rows: list[dict[str, object]] = []
        rejection_rows: list[dict[str, object]] = []

        split_sources = {
            "train": train_sources,
            "validation": validation_sources,
            "test": test_files,
        }
        for split, files in split_sources.items():
            for path in files:
                values = base.load_dat(path)
                valid, recomputed, reason = base.evaluate_geometry(
                    values,
                    args.n_points_per_side,
                    args.descriptor_method,
                    args.uniform_points,
                    args.geometry_tolerance,
                )
                source_rows.append(
                    {
                        "filename": path.name,
                        "source_id": path.stem,
                        "split": split,
                        "sha256": base.sha256(path),
                        "stored_max_thickness": values[16],
                        "stored_max_thickness_position": values[17],
                        "stored_max_camber": values[18],
                        "stored_max_camber_position": values[19],
                        "recomputed_geometry_valid": valid,
                        "recomputed_geometry_status": reason,
                        "recomputed_max_thickness": (
                            "" if recomputed is None else recomputed[0]
                        ),
                        "recomputed_max_thickness_position": (
                            "" if recomputed is None else recomputed[1]
                        ),
                        "recomputed_max_camber": (
                            "" if recomputed is None else recomputed[2]
                        ),
                        "recomputed_max_camber_position": (
                            "" if recomputed is None else recomputed[3]
                        ),
                    }
                )

        # Validation and test remain source-only and byte-identical.
        for path in validation_sources:
            base.copy_original(path, validation_output_dir)
        for path in test_files:
            base.copy_original(path, test_output_dir)

        total_rejections = 0
        rejection_reason_counts: dict[str, int] = defaultdict(int)

        for source_number, source_path in enumerate(train_sources, start=1):
            source_values = base.load_dat(source_path)
            copied_original = base.copy_original(source_path, train_output_dir)
            rng, derived_seed = base.source_rng(args.seed, source_path.name)

            train_rows.append(
                {
                    "output_filename": copied_original.name,
                    "source_filename": source_path.name,
                    "source_id": source_path.stem,
                    "kind": "original",
                    "augmentation_index": 0,
                    "source_rng_seed": derived_seed,
                    "accepted_attempt_number": 0,
                    "absolute_offsets_dim_1_to_14": "",
                    "max_thickness": source_values[16],
                    "max_thickness_position": source_values[17],
                    "max_camber": source_values[18],
                    "max_camber_position": source_values[19],
                    "sha256": base.sha256(copied_original),
                }
            )

            accepted = 0
            attempts = 0
            source_rejections: dict[str, int] = defaultdict(int)

            while accepted < args.augmentations_per_source:
                attempts += 1
                if attempts > args.max_attempts_per_source:
                    raise RuntimeError(
                        f"{source_path.name}: generated only {accepted} valid "
                        f"samples after {args.max_attempts_per_source} attempts"
                    )

                # Absolute additive perturbation; no percentage scaling.
                offsets = rng.uniform(
                    -args.absolute_perturbation,
                    args.absolute_perturbation,
                    size=14,
                )
                candidate = source_values.copy()
                candidate[:14] = source_values[:14] + offsets
                candidate[14:16] = source_values[14:16]

                # Match this release: evaluate descriptors before %.8f serialization.

                valid, descriptors, reason = base.evaluate_geometry(
                    candidate,
                    args.n_points_per_side,
                    args.descriptor_method,
                    args.uniform_points,
                    args.geometry_tolerance,
                )
                if not valid or descriptors is None:
                    total_rejections += 1
                    source_rejections[reason] += 1
                    rejection_reason_counts[reason] += 1
                    continue

                candidate[16:20] = descriptors
                accepted += 1
                output_name = f"{source_path.stem}__abs005_aug_{accepted:02d}.dat"
                output_path = train_output_dir / output_name
                if output_path.exists():
                    raise FileExistsError(f"Output filename collision: {output_name}")
                base.save_dat(output_path, candidate)

                # Reopen and validate after rounding to eight decimals.
                saved = base.load_dat(output_path)
                if not np.array_equal(saved[14:16], source_values[14:16]):
                    raise RuntimeError(f"{output_name}: dimensions 15-16 changed")
                saved_valid, saved_descriptors, saved_reason = base.evaluate_geometry(
                    saved,
                    args.n_points_per_side,
                    args.descriptor_method,
                    args.uniform_points,
                    args.geometry_tolerance,
                )
                if not saved_valid or saved_descriptors is None:
                    raise RuntimeError(
                        f"{output_name}: invalid after saving: {saved_reason}"
                    )
                if not np.allclose(
                    saved[16:20], saved_descriptors, atol=1e-6, rtol=0.0
                ):
                    raise RuntimeError(
                        f"{output_name}: stored descriptors failed verification"
                    )

                train_rows.append(
                    {
                        "output_filename": output_name,
                        "source_filename": source_path.name,
                        "source_id": source_path.stem,
                        "kind": "augmented",
                        "augmentation_index": accepted,
                        "source_rng_seed": derived_seed,
                        "accepted_attempt_number": attempts,
                        "absolute_offsets_dim_1_to_14": ";".join(
                            f"{value:.10f}" for value in offsets
                        ),
                        "max_thickness": saved[16],
                        "max_thickness_position": saved[17],
                        "max_camber": saved[18],
                        "max_camber_position": saved[19],
                        "sha256": base.sha256(output_path),
                    }
                )

            rejection_rows.append(
                {
                    "source_filename": source_path.name,
                    "source_id": source_path.stem,
                    "attempt_count": attempts,
                    "accepted_count": accepted,
                    "rejected_count": attempts - accepted,
                    "rejection_reasons": json.dumps(
                        source_rejections, ensure_ascii=False, sort_keys=True
                    ),
                }
            )

            if source_number % 50 == 0 or source_number == len(train_sources):
                print(
                    f"Processed {source_number}/{len(train_sources)} train sources; "
                    f"total rejected attempts: {total_rejections}",
                    flush=True,
                )

        source_fields = [
            "filename",
            "source_id",
            "split",
            "sha256",
            "stored_max_thickness",
            "stored_max_thickness_position",
            "stored_max_camber",
            "stored_max_camber_position",
            "recomputed_geometry_valid",
            "recomputed_geometry_status",
            "recomputed_max_thickness",
            "recomputed_max_thickness_position",
            "recomputed_max_camber",
            "recomputed_max_camber_position",
        ]
        train_fields = [
            "output_filename",
            "source_filename",
            "source_id",
            "kind",
            "augmentation_index",
            "source_rng_seed",
            "accepted_attempt_number",
            "absolute_offsets_dim_1_to_14",
            *base.DESCRIPTOR_NAMES,
            "sha256",
        ]
        print("Writing dataset manifests...", flush=True)
        base.write_csv(
            manifests_dir / "source_split_manifest.csv", source_fields, source_rows
        )
        base.write_csv(
            manifests_dir / "train_files_manifest.csv", train_fields, train_rows
        )
        base.write_csv(
            manifests_dir / "augmentation_rejections.csv",
            [
                "source_filename",
                "source_id",
                "attempt_count",
                "accepted_count",
                "rejected_count",
                "rejection_reasons",
            ],
            rejection_rows,
        )

        print("Checking final file counts...", flush=True)
        actual_train_files = len(list(train_output_dir.glob("*.dat")))
        actual_validation_files = len(list(validation_output_dir.glob("*.dat")))
        actual_test_files = len(list(test_output_dir.glob("*.dat")))
        if actual_train_files != planned_train_files:
            raise RuntimeError(
                f"Expected {planned_train_files} train files, "
                f"found {actual_train_files}"
            )
        if actual_validation_files != len(validation_sources):
            raise RuntimeError("Validation output file count mismatch")
        if actual_test_files != len(test_files):
            raise RuntimeError("Test output file count mismatch")
        if len(train_rows) != actual_train_files:
            raise RuntimeError("Train manifest row count mismatch")

        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **plan,
            "python_version": os.sys.version.split()[0],
            "aerosandbox_version": asb.__version__,
            "numpy_version": np.__version__,
            "perturbed_dimensions_one_based": list(range(1, 15)),
            "unchanged_dimensions_one_based": [15, 16],
            "recomputed_dimensions_one_based": [17, 18, 19, 20],
            "perturbation_formula": (
                "new[i] = source[i] + Uniform(-absolute_perturbation, "
                "+absolute_perturbation)"
            ),
            "coordinate_method": {
                "function": "get_kulfan_coordinates",
                "N1": 0.5,
                "N2": 1.0,
                "n_points_per_side": args.n_points_per_side,
                "TE_thickness_policy": "max(0, dimension_15)",
            },
            "original_train_samples_policy": (
                "copied unchanged into train_combined"
            ),
            "validation_policy": "original files only; no augmentation",
            "test_policy": "original files only; no augmentation",
            "geometry_validity_policy": (
                "finite Kulfan coordinates, monotonic surface x coordinates, "
                "and vertical thickness >= -geometry_tolerance"
            ),
            "geometry_tolerance": args.geometry_tolerance,
            "total_rejected_attempts": total_rejections,
            "rejection_reason_counts": dict(
                sorted(rejection_reason_counts.items())
            ),
            "actual_train_file_count": actual_train_files,
            "actual_validation_file_count": actual_validation_files,
            "actual_test_file_count": actual_test_files,
        }
        (manifests_dir / "dataset_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        readme = f"""# R1 source-level dataset: absolute CST perturbation

Generated by `augment_training_airfoils_absolute.py`.

- Train sources: {len(train_sources)}
- Validation sources: {len(validation_sources)}
- Test sources: {len(test_files)}
- Augmented variants per train source: {args.augmentations_per_source}
- Total files in `train_combined`: {actual_train_files}
- Perturbation: independent additive Uniform(-{args.absolute_perturbation:.4f}, +{args.absolute_perturbation:.4f}) on dimensions 1-14
- Dimensions 15-16: unchanged
- Dimensions 17-20: recomputed using `{args.descriptor_method}`
- AeroSandbox version: {asb.__version__}
- Random seed: {args.seed}

Original training files and augmented files are stored together in
`train_combined`. Validation and test files remain unaugmented. See the CSV
and JSON files in `manifests` for exact provenance and rejection statistics.
"""
        (temp_root / "README.md").write_text(readme, encoding="utf-8")

    except Exception as exc:
        print(f"Dataset generation failed: {exc}", flush=True)
        if temp_root.exists() and temp_root.parent == output_root.parent:
            print(f"Removing incomplete dataset: {temp_root}", flush=True)
            shutil.rmtree(temp_root)
        raise

    # Finalization is outside the cleanup block: a rename failure must never
    # delete a completed and verified dataset.
    print("Dataset verified; finalizing output directory...", flush=True)
    try:
        finalize_output(temp_root, output_root)
    except OSError:
        print(
            f"Finalization failed. Complete dataset preserved at: {temp_root}\n"
            f"After resolving the directory access issue, rename it to: {output_root}\n"
            "No regeneration is needed for this preserved dataset.",
            flush=True,
        )
        raise

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
