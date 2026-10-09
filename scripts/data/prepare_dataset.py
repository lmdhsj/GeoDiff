"""Verify/extract the released data ZIP or compare regenerated dataset records."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import stat
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[2]
NAME = "cst_airfoil_dim_20_source_split_r1_aug20_abs005"
ARCHIVE = REPO / "dataset/geodiff_review_r2_abs005_dataset.zip"

def rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))

def expected_records(z):
    prefix = f"dataset/{NAME}/"
    sources = rows(z.read(prefix + "manifests/source_split_manifest.csv"))
    training = rows(z.read(prefix + "manifests/train_files_manifest.csv"))
    expected = {}
    for row in training:
        key = "train_combined/" + row["output_filename"]
        if key in expected:
            raise ValueError("Duplicate training manifest filename")
        expected[key] = row["sha256"]
    for row in sources:
        if row["split"] != "train":
            key = row["split"] + "_original/" + row["filename"]
            if key in expected:
                raise ValueError("Duplicate source manifest filename")
            expected[key] = row["sha256"]
    if Counter(r["split"] for r in sources) != {"train": 1080, "validation": 120, "test": 300}:
        raise ValueError("Wrong source split counts")
    if Counter(r["kind"] for r in training) != {"original": 1080, "augmented": 21600}:
        raise ValueError("Wrong training counts")
    return sources, training, expected

def verify_archive(archive):
    sums = archive.parent / "SHA256SUMS.txt"
    expected_zip_hash = None
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, filename = line.split(maxsplit=1)
            if filename.lstrip("*") == archive.name:
                expected_zip_hash = digest
    if expected_zip_hash is None or hashlib.sha256(archive.read_bytes()).hexdigest() != expected_zip_hash:
        raise ValueError("Data ZIP checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        for item in z.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename or ":" in item.filename:
                raise ValueError("Unsafe archive path")
            if not item.filename.startswith("dataset/") or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unexpected archive entry")
        sources, training, expected = expected_records(z)
        prefix = f"dataset/{NAME}/"
        actual = {n.removeprefix(prefix) for n in names if n.startswith(prefix) and n.endswith(".dat")}
        if actual != set(expected):
            raise ValueError("Dataset file list differs from manifests")
        original_prefix = "dataset/cst_airfoil_dim_20/"
        original_names = {n.removeprefix(original_prefix) for n in names if n.startswith(original_prefix) and n.endswith(".dat")}
        if original_names != {r["filename"] for r in sources}:
            raise ValueError("Wrong original source file list")
        original_values = {}
        for row in sources:
            raw = z.read(original_prefix + row["filename"])
            values = [float(v) for v in raw.decode("utf-8-sig").split()]
            if len(values) != 20 or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid original parameter record")
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Original source hash mismatch: " + row["filename"])
            original_values[row["filename"]] = values
            directory = "train_combined" if row["split"] == "train" else row["split"] + "_original"
            if z.read(prefix + directory + "/" + row["filename"]) != raw:
                raise ValueError("Original source changed in final dataset")
        minimum, maximum = float("inf"), float("-inf")
        for key, digest in expected.items():
            raw = z.read(prefix + key)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Record hash mismatch: " + key)
            values = [float(v) for v in raw.decode("utf-8-sig").split()]
            if len(values) != 20 or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid parameter record: " + key)
        augmented_counts = Counter()
        for row in training:
            if row["kind"] == "augmented":
                values = [float(v) for v in z.read(prefix + "train_combined/" + row["output_filename"]).decode().split()]
                original = original_values[row["source_filename"]]
                offsets = [values[i] - original[i] for i in range(14)]
                if max(abs(d) for d in offsets) > 0.05 + 1e-8 or values[14:16] != original[14:16]:
                    raise ValueError("Augmentation rule mismatch: " + row["output_filename"])
                minimum, maximum = min(minimum, min(offsets)), max(maximum, max(offsets))
                augmented_counts[row["source_filename"]] += 1
        if len(augmented_counts) != 1080 or set(augmented_counts.values()) != {20}:
            raise ValueError("Wrong augmented variant count per source")
        split_hashes = {"train": set(), "validation": set(), "test": set()}
        for row in sources:
            split_hashes[row["split"]].add(row["sha256"])
        split_hashes["train"].update(r["sha256"] for r in training)
        if any(split_hashes[a] & split_hashes[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
            raise ValueError("Identical data records cross split boundaries")
        metadata = json.loads(z.read(prefix + "manifests/dataset_metadata.json"))
        if metadata["seed"] != 44 or metadata["absolute_perturbation"] != 0.05 or metadata["augmentations_per_train_source"] != 20:
            raise ValueError("Wrong augmentation protocol metadata")
        bad_entry = z.testzip()
        if bad_entry is not None:
            raise ValueError("ZIP integrity error: " + bad_entry)
    return {"status": "passed", "archive_sha256": expected_zip_hash, "original_sources": 1500,
            "original_training_sources": 1080, "augmented_training_variants": 21600,
            "training_samples": 22680, "validation_samples": 120, "test_samples": 300,
            "total_final_dataset_records": len(expected), "absolute_offset_min": minimum,
            "absolute_offset_max": maximum, "cross_split_identical_hashes": 0,
            "original_records_byte_identical": True, "all_data_file_hashes_match": True}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-archive-only", action="store_true")
    parser.add_argument("--compare-reproduced", type=Path)
    args = parser.parse_args()
    if args.compare_reproduced:
        with zipfile.ZipFile(ARCHIVE) as z:
            _, _, expected = expected_records(z)
        actual = {p.relative_to(args.compare_reproduced).as_posix() for p in args.compare_reproduced.rglob("*.dat")}
        mismatches = sorted(actual ^ set(expected))
        def compare_one(name):
            digest = hashlib.sha256((args.compare_reproduced / name).read_bytes()).hexdigest()
            return name if digest != expected[name] else None
        with ThreadPoolExecutor(max_workers=16) as pool:
            mismatches.extend(name for name in pool.map(compare_one, sorted(actual & set(expected))) if name is not None)
        print(json.dumps({"compared_records": len(expected), "mismatch_count": len(mismatches), "first_mismatches": mismatches[:20]}, indent=2))
        if mismatches:
            raise SystemExit(1)
        return
    report = verify_archive(ARCHIVE)
    print(json.dumps(report, indent=2), flush=True)
    if args.verify_archive_only:
        return
    destinations = [REPO / "dataset" / name for name in [NAME, "cst_airfoil_dim_20", "source_split_r1_manifests"]]
    if any(p.exists() for p in destinations):
        raise FileExistsError("A dataset destination already exists. No files were overwritten; use --verify-archive-only to verify the release ZIP.")
    # Stage extraction in a temporary sibling before moving complete directories.
    with tempfile.TemporaryDirectory(prefix="geodiff-extract-", dir=REPO) as staging:
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(staging)
        for destination in destinations:
            if destination.exists():
                raise FileExistsError(str(destination))
            shutil.move(str(Path(staging) / "dataset" / destination.name), str(destination))
    print("Extracted dataset under " + str(REPO / "dataset"))

if __name__ == "__main__":
    main()
