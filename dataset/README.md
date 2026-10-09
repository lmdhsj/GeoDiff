# GeoDiff dataset files

Download `geodiff_abs005_dataset.zip` and run `python scripts/data/prepare_dataset.py` from the repository root. The ZIP contains both the 1,500 processed originals and the final training/validation/test dataset. See [the dataset guide](../DATASET.md) for acquisition, schema, and reproduction commands.

The augmentation uses **absolute additive offsets in [-0.05, +0.05]**, with 20 valid variants per training source. It is not a relative percentage perturbation.
