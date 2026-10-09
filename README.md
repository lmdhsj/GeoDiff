# GeoDiff

GeoDiff generates airfoils from four geometric conditions using a conditional diffusion model in an autoencoder latent space. The current `review_r2` configuration uses a **20-dimensional airfoil representation** and an **8-dimensional latent space**.

This release provides the processed dataset used by the current model, the original processed airfoils, exact split manifests, and scripts for reproducing the source split and training-only augmentation.

## Dataset download

Download [`dataset/geodiff_review_r2_abs005_dataset.zip`](dataset/geodiff_review_r2_abs005_dataset.zip), or clone this repository:

```bash
git clone https://github.com/lmdhsj/GeoDiff.git
cd GeoDiff
python scripts/data/prepare_dataset.py
```

The preparation command verifies the archive and extracts the dataset. It requires only the Python standard library. Existing dataset directories are never overwritten. To verify the ZIP without extracting it:

```bash
python scripts/data/prepare_dataset.py --verify-archive-only
```

## Dataset composition

| Subset | Original airfoils | Augmented variants | Total samples |
| --- | ---: | ---: | ---: |
| Training set | 1,080 | 21,600 | 22,680 |
| Validation set | 120 | 0 | 120 |
| Test set | 300 | 0 | 300 |

The 1,500 original processed airfoils were first divided into 1,200 development airfoils and 300 test airfoils. Of the development airfoils, 120 were reserved for validation, leaving 1,080 for training. The source-level splits use seed **44**, with byte-identical records kept in the same subset.

Training uses the augmented dataset. Evaluation on the **original training set** uses only the 1,080 original records, selected using `kind == "original"` in `train_files_manifest.csv`. The validation and test sets contain original records only.

## Training data augmentation

For each training airfoil, independent additive offsets were applied to the first 14 CST coefficients:

```text
a_aug[i] = a_original[i] + epsilon[i]
epsilon[i] ~ Uniform(-0.05, +0.05), i = 0, ..., 13
```

**The amplitude is an absolute coefficient offset, not a percentage.** The trailing-edge thickness and leading-edge-modification weight were retained. The four geometric descriptors were recomputed for each augmented sample using 100 cosine-spaced nodes per surface in AeroSandbox's CST-with-LEM representation.

Candidates were checked for finite coordinates, monotonic surface x coordinates, and nonnegative thickness at the sampled nodes, with tolerance `1e-8`. Invalid candidates were rejected and resampled until 20 valid variants were obtained per training source. The original training records were also retained. There were 306 rejected attempts in this release; their counts and reasons are recorded in the manifests.

## Data format

Each processed `.dat` file contains **20 whitespace-separated numeric values**, rather than an x-y coordinate table:

| Python index | Field | Count |
| --- | --- | ---: |
| `[0:7]` | Upper-surface CST coefficients | 7 |
| `[7:14]` | Lower-surface CST coefficients | 7 |
| `[14]` | Trailing-edge thickness | 1 |
| `[15]` | Leading-edge-modification weight | 1 |
| `[16]` | Maximum thickness | 1 |
| `[17]` | Maximum thickness position | 1 |
| `[18]` | Maximum camber | 1 |
| `[19]` | Maximum camber position | 1 |

Thickness and camber are expressed relative to chord length; positions are expressed as `x/c`. The four GeoDiff conditions are **`parameters[16:20]`**, in the order shown above. The AE uses all 20 values.

## Original data source and reproducibility

The airfoils originate from the [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html). Raw coordinate files can be obtained from the [official Selig-format ZIP archive](https://m-selig.ae.illinois.edu/ads/archives/coord_seligFmt.zip).

The original coordinate preprocessing includes interpolation and CST parameterization using [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox). This release contains the **exact 1,500 processed records** used by the project. Here, “original” means before training augmentation; these records have already been preprocessed and parameterized. The current complete UIUC archive can differ from the fixed set used in this study.

The supplied reproduction scripts begin with the released 1,500 processed records. The historical raw-coordinate interpolation/fitting export script is not included in this release. See [`DATASET.md`](DATASET.md) for the folder structure, exact reproduction commands, coordinate reconstruction, and manifest details.

The published split and augmentation scripts were run to regenerate all **23,100 final dataset records** (training, validation, and test), and every regenerated `.dat` file matched the released record by SHA-256. See [archive verification](dataset/verification.json) and [full reproduction verification](dataset/reproduction_verification.json).

Model training code and checkpoint files are outside the scope of this dataset release.
