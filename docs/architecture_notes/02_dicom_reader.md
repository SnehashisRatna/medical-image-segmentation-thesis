# DICOM Reader

## 1. Overview

The DICOM Reader is the core input component of the segmentation framework. It abstracts all interactions with DICOM files and provides a unified interface for reading both image data and acquisition metadata.

Instead of allowing multiple modules to directly use pydicom, all DICOM operations are centralized within this module. This design improves maintainability, reduces code duplication, and simplifies future extensions.

The DICOM Reader acts as the gateway between raw medical images and the remaining components of the framework.

---

## 2. Motivation

DICOM files contain much more than image pixels.

They also include:

- Patient information
- Acquisition parameters
- Pixel spacing
- Slice thickness
- Rescale coefficients
- Scanner information

Allowing every module to directly access pydicom would lead to duplicated code and inconsistent metadata handling.

Therefore, a dedicated DICOM Reader was developed.

---

## 3. Research Context

Medical imaging frameworks such as MONAI and ITK isolate DICOM handling into dedicated components.

Inspired by this architecture, our framework introduces a reusable DICOM Reader that separates image reading from preprocessing, metadata analysis, and dataset loading.

This abstraction improves reproducibility and simplifies future maintenance.

---

## 4. Design Goals

The module was designed to:

- Read DICOM images
- Read DICOM metadata
- Convert pixel data to NumPy arrays
- Compute image statistics
- Hide pydicom implementation details
- Provide a consistent API
- Support future datasets

---

## 5. Architecture

Raw DICOM File

        │

        ▼

DICOM Reader

        │

        ├── read_metadata()

        ├── read_image()

        └── image_statistics()

        │

        ▼

Dataset Explorer

Metadata Scanner

Future Dataset Loader

---

## 6. Design Decisions

### Why isolate DICOM operations?

Without a dedicated reader:

Dataset Explorer

↓

pydicom

Metadata Scanner

↓

pydicom

Dataset Loader

↓

pydicom

This duplicates logic and tightly couples every module to pydicom.

Instead:

Dataset Explorer

↓

DICOM Reader

↓

pydicom

Metadata Scanner

↓

DICOM Reader

↓

pydicom

Only one module knows how DICOM files are handled.

### Why return structured metadata?

Returning metadata through a dedicated interface makes it easier to later replace dictionaries with a DICOMMetadata dataclass without affecting higher-level modules.

---

## 7. Implementation Details

Main methods:

- read_metadata()
- read_image()
- image_statistics()

The reader extracts acquisition metadata and converts pixel data into NumPy arrays suitable for preprocessing and visualization.

---

## 8. Validation

The module successfully verified:

✓ Metadata extraction

✓ Image loading

✓ Pixel statistics

✓ Integration with Dataset Explorer

✓ Integration with Metadata Scanner

---

## 9. Research Findings

Metadata analysis revealed:

- CT uses scanner-dependent rescale intercept values.
- MRI sequences use different slice thicknesses.
- Pixel spacing varies across studies.
- Image intensities require modality-specific preprocessing.

These findings directly influenced the design of the preprocessing pipeline.

---

## 10. Future Improvements

- Introduce DICOMMetadata dataclass
- Add exception handling
- Support compressed DICOM files
- Logging
- Parallel reading
- Volume reconstruction support

---

## 11. Contribution to Thesis

The DICOM Reader establishes a reusable and consistent interface for accessing medical imaging data throughout the segmentation framework.

It provides the technical foundation required for preprocessing, visualization, dataset loading, and model training.

---

## 12. Contribution to Future Paper

The modular DICOM Reader enables reproducible data access across multiple medical imaging datasets and can be extended for future multimodal segmentation research.