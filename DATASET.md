# Dataset acquisition and reproduction

## Obtain the data

For the exact dataset used by the current GeoDiff model, download `dataset/geodiff_abs005_dataset.zip` from this repository and run:

```bash
python scripts/data/prepare_dataset.py
```

The ZIP includes both the final model dataset and the 1,500 original processed parameter files:

```text
dataset/
  cst_airfoil_dim_20/                           # 1,500 processed originals
  cst_airfoil_dim_20_source_split_r1_aug20_abs005/
    train_combined/                           # 22,680 .dat files
    validation_original/                      # 120 .dat files
    test_original/                            # 300 .dat files
    manifests/
      source_split_manifest.csv
      train_files_manifest.csv
      augmentation_rejections.csv
      dataset_metadata.json
    README.md
  source_split_r1_manifests/                   # historical 1,200/300 split records
```

Readable copies of the final source split and protocol metadata are also available directly in `dataset/manifests/` on GitHub. The full augmented-sample provenance manifest is inside the ZIP.

Raw UIUC coordinate files are available separately from the [official database](https://m-selig.ae.illinois.edu/ads/coord_database.html) and its [Selig-format ZIP](https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip). Raw UIUC `.dat` files contain coordinate data; the processed `.dat` files in this release contain 20 parameters. They must not be loaded as the same format.

Original coordinate interpolation and CST fitting are distinct from augmentation descriptor evaluation. The initial raw-to-parameter export program is not supplied; exact split/augmentation reproduction starts from the processed originals distributed here. The latest UIUC archive is not guaranteed to reproduce the project's fixed 1,500 processed records.

## Reproduce the split and augmentation

Use Python 3.12. The original augmentation metadata records Python 3.12.9, NumPy 1.26.4, and AeroSandbox 4.2.8. Install the data preparation dependencies:

```bash
python -m pip install -r requirements-data.txt
```

Run these commands from the repository root after extracting the released archive. Use a new output directory; the scripts refuse to overwrite existing data.

```bash
python scripts/data/split_source_airfoils.py --source-dir dataset/cst_airfoil_dim_20 --output-dir dataset/reproduced/source_split_r1 --seed 44 --test-ratio 0.2 --expected-count 1500
python scripts/data/augment_training_airfoils_absolute.py --input-root dataset/reproduced/source_split_r1 --output-root dataset/reproduced/source_split_r1_aug20_abs005 --seed 44 --validation-ratio 0.10 --augmentations-per-source 20 --absolute-perturbation 0.05 --n-points-per-side 100 --descriptor-method kulfan_nodes --geometry-tolerance 1e-8
```

The first command produces 1,200 development files in `train_original` and 300 test files in `test_original`. The second command splits development sources into 1,080 training and 120 validation sources, retains the original training records, and generates 20 accepted variants per training source.

The split is deterministic, grouping files by SHA-256 and selecting an exact subset file count using seed 44. There are 1,486 unique byte-content groups among the 1,500 original files. The 14 duplicate-content groups are kept together. No identical original record crosses the training, validation, or test boundaries.

Random augmentation uses a separate generator for each original filename. Its seed is the little-endian integer from the first eight bytes of `SHA256("44:" + filename)`. This makes the augmentation sequence reproducible for each source. Accepted-attempt indices and sampled offsets are included in the training manifest.

Descriptors in this release are evaluated from the perturbed CST parameters before final file serialization. Outputs use `%.8f` precision. Consequently, recomputing descriptors from the rounded saved coefficients can produce small differences in the last decimal place. The saved descriptor check uses absolute tolerance `1e-6`. Descriptors are maxima of sampled thickness and camber, and the x/c coordinates of those maxima. The lower surface is interpolated to the upper-surface node locations with linear interpolation. Original files, including their stored descriptors, are copied unchanged.

For a byte-level comparison of regenerated data with the released records:

```bash
python scripts/data/prepare_dataset.py --compare-reproduced dataset/reproduced/source_split_r1_aug20_abs005
```

This compares every `.dat` file against the release hash manifest and reports mismatches. Metadata creation times and output paths naturally differ between runs. Exact numerical results can also depend on the software environment; use the released files and their hashes as the fixed reference.

## Load original training airfoils

Original files and augmented files coexist in `train_combined`. Select originals using the manifest, rather than selecting arbitrary filenames or assuming all training files are original:

```python
import csv
from pathlib import Path
import numpy as np

root = Path("dataset/cst_airfoil_dim_20_source_split_r1_aug20_abs005")
with (root / "manifests/train_files_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
    originals = [row["output_filename"] for row in csv.DictReader(f)
                 if row["kind"] == "original"]
assert len(originals) == 1080
parameters = np.stack([np.loadtxt(root / "train_combined" / name).reshape(20)
                       for name in originals])
conditions = parameters[:, 16:20]
```

## Reconstruct a coordinate contour

```python
import numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates

p = np.loadtxt("dataset/cst_airfoil_dim_20/naca4412.dat").reshape(20)
coordinates = get_kulfan_coordinates(
    upper_weights=p[:7], lower_weights=p[7:14],
    TE_thickness=max(0.0, float(p[14])), leading_edge_weight=float(p[15]),
    N1=0.5, N2=1.0, n_points_per_side=100,
)
```

Each surface has 100 cosine-spaced nodes. The shared leading-edge point appears only once in the joined contour, giving 199 contour entries. This is the augmentation geometry/descriptor convention. Coordinate RMSE evaluation can use an additional interpolation grid; it is a separate operation.

## Manifest contents

| File | Purpose |
| --- | --- |
| `source_split_manifest.csv` | Exact membership and hashes of all 1,500 original sources; stored and recomputed descriptors |
| `train_files_manifest.csv` | All 22,680 training records; original/augmented flag, source identity, RNG seed, accepted attempt, offsets, descriptors, hashes |
| `augmentation_rejections.csv` | Attempt and rejection counts per training source; 306 total rejected attempts |
| `dataset_metadata.json` | Dataset counts, seed, augmentation amplitude, numerical settings, dependency versions |
| `source_split_r1_manifests/` | Reference manifests for the initial 1,200/300 source split |
| `dataset/SHA256SUMS.txt` | Checksum of the downloadable data ZIP |
| `dataset/verification.json` | Verification performed when preparing this release |

The release preserves all `.dat` bytes and CSV provenance records. Only local input/output directory fields in the two metadata JSON files have been replaced with relative paths for portability. The data ZIP stores files under `dataset/`; do not extract it into another nested `dataset/` folder.

Absolute CST augmentation offsets are independent of relative bounds used on geometric conditions during optimization. The training augmentation amplitude must not be described as ±20%.
