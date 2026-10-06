from pathlib import Path

import numpy as np
import pydicom


ROOT = Path("data/raw/CHAOS_Train_Sets/Train_Sets/MR")


sequences = {
    "T1DUAL_InPhase": (
        "T1DUAL",
        "DICOM_anon",
        "InPhase",
    ),
    "T1DUAL_OutPhase": (
        "T1DUAL",
        "DICOM_anon",
        "OutPhase",
    ),
    "T2SPIR": (
        "T2SPIR",
        "DICOM_anon",
    ),
}


patients = sorted(
    [p for p in ROOT.iterdir() if p.is_dir()],
    key=lambda p: int(p.name),
)


for sequence_name, folders in sequences.items():
    print()
    print("=" * 60)
    print(sequence_name)
    print("=" * 60)

    files = []

    for patient in patients:
        directory = patient.joinpath(*folders)

        if directory.exists():
            files.extend(sorted(directory.glob("*.dcm")))

    print(f"DICOM files: {len(files)}")

    if not files:
        continue

    mins = []
    maxs = []
    means = []
    stds = []

    nonzero_mins = []
    nonzero_maxs = []
    nonzero_means = []
    nonzero_stds = []

    percentiles = []

    for path in files:
        ds = pydicom.dcmread(path)

        image = ds.pixel_array.astype(np.float32)

        mins.append(float(image.min()))
        maxs.append(float(image.max()))
        means.append(float(image.mean()))
        stds.append(float(image.std()))

        nonzero = image[image > 0]

        if nonzero.size > 0:
            nonzero_mins.append(float(nonzero.min()))
            nonzero_maxs.append(float(nonzero.max()))
            nonzero_means.append(float(nonzero.mean()))
            nonzero_stds.append(float(nonzero.std()))

            percentiles.append(
                np.percentile(
                    nonzero,
                    [1, 5, 50, 95, 99],
                )
            )

    percentiles = np.asarray(percentiles)

    print(f"Global min: {min(mins):.2f}")
    print(f"Global max: {max(maxs):.2f}")

    print(f"Mean of slice means: {np.mean(means):.2f}")
    print(f"Mean of slice stds: {np.mean(stds):.2f}")

    print(f"Median slice mean: {np.median(means):.2f}")
    print(f"Median slice std: {np.median(stds):.2f}")

    print()
    print("--- Non-zero pixels ---")

    print(f"Global non-zero min: {min(nonzero_mins):.2f}")
    print(f"Global non-zero max: {max(nonzero_maxs):.2f}")

    print(f"Mean non-zero slice mean: {np.mean(nonzero_means):.2f}")
    print(f"Mean non-zero slice std: {np.mean(nonzero_stds):.2f}")

    print()
    print("--- Non-zero percentile distribution ---")

    print(
        f"1st percentile:  {np.median(percentiles[:, 0]):.2f}"
    )
    print(
        f"5th percentile:  {np.median(percentiles[:, 1]):.2f}"
    )
    print(
        f"50th percentile: {np.median(percentiles[:, 2]):.2f}"
    )
    print(
        f"95th percentile: {np.median(percentiles[:, 3]):.2f}"
    )
    print(
        f"99th percentile: {np.median(percentiles[:, 4]):.2f}"
    )