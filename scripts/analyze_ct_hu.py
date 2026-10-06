from pathlib import Path

import numpy as np
import pydicom


ROOT = Path("data/raw/CHAOS_Train_Sets/Train_Sets/CT")

mins = []
maxs = []
means = []
stds = []

files = sorted(ROOT.glob("*/DICOM_anon/*.dcm"))

print(f"DICOM files: {len(files)}")

for path in files:
    ds = pydicom.dcmread(path)

    raw = ds.pixel_array.astype(np.float32)

    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    hu = raw * slope + intercept

    mins.append(float(hu.min()))
    maxs.append(float(hu.max()))
    means.append(float(hu.mean()))
    stds.append(float(hu.std()))

print(f"Global min HU: {min(mins):.2f}")
print(f"Global max HU: {max(maxs):.2f}")
print(f"Mean of slice means: {np.mean(means):.2f}")
print(f"Mean of slice stds: {np.mean(stds):.2f}")
print(f"Median slice mean HU: {np.median(means):.2f}")
print(f"Median slice std HU: {np.median(stds):.2f}")