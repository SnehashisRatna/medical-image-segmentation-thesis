# Dataset Explorer

## 1. Overview

The Dataset Explorer is the first module developed in the Medical Image Segmentation Framework. Its primary objective is to inspect the CHAOS dataset, verify its directory structure, and ensure that all imaging studies are correctly organized before any preprocessing or model development begins.

Unlike a conventional script that simply counts files, this module acts as an initial validation layer for the complete research pipeline. It provides confidence that the dataset has been downloaded correctly and that subsequent modules can safely assume a valid dataset structure.

This module is intentionally lightweight, focusing only on dataset organization and basic visualization rather than detailed metadata extraction.

---

## 2. Motivation

Medical image datasets frequently contain hundreds or thousands of DICOM slices organized in nested directory structures. Any inconsistency in folder organization may lead to failures during preprocessing or model training.

Instead of discovering structural issues during training, the Dataset Explorer performs an early verification of the dataset. This reduces debugging effort and provides an initial understanding of the data distribution.

For the CHAOS dataset, both CT and MRI studies use different directory layouts. Therefore, a dedicated exploration module was developed before implementing preprocessing and segmentation models.

---

## 3. Research Context

Most modern medical imaging frameworks begin by validating dataset integrity before processing images.

Frameworks such as:

- MONAI
- nnU-Net
- TotalSegmentator

perform dataset inspection before preprocessing.

Inspired by this workflow, the Dataset Explorer was designed as the first stage of our segmentation framework.

---

## 4. Design Goals

The module was designed with the following objectives:

- Automatically verify dataset structure
- Count CT and MRI patients
- Confirm dataset completeness
- Visualize sample images
- Keep implementation lightweight
- Avoid preprocessing responsibilities
- Provide a foundation for later modules

---

## 5. Architecture

CHAOS Dataset

        │

        ▼

Dataset Explorer

        │

        ├── Count CT Patients

        ├── Count MR Patients

        ├── Display Dataset Summary

        └── Visualize Sample Slice

---

## 6. Design Decisions

### Why create a separate Dataset Explorer?

Instead of embedding exploration inside the Dataset Loader, dataset validation was isolated into an independent module.

Advantages:

- Easier debugging
- Clear separation of responsibilities
- Faster verification
- Reusable for new datasets

### Why not perform preprocessing here?

Preprocessing modifies image intensities and geometry.

Dataset exploration should only inspect the data.

Keeping these responsibilities separate follows the Single Responsibility Principle.

---

## 7. Implementation Details

Main methods:

- count_patients()
- print_summary()
- read_first_ct_metadata()
- visualize_first_ct_slice()

The explorer uses DICOMReader for reading DICOM files rather than directly accessing pydicom.

---

## 8. Validation

The module successfully verified:

✓ CT Patients = 20

✓ MRI Patients = 20

✓ Total Patients = 40

✓ CT image visualization

✓ Dataset folder integrity

---

## 9. Research Findings

During dataset exploration we observed:

- CT images are stored as 512×512 DICOM slices.
- MRI images are organized into T1DUAL and T2SPIR sequences.
- CT and MRI follow different directory structures.
- The dataset is suitable for multimodal segmentation research.

These observations motivated the modular design adopted throughout the framework.

---

## 10. Future Improvements

- Display random patient samples
- Interactive visualization
- Dataset statistics dashboard
- Support additional medical imaging datasets
- Integration with future preprocessing pipeline

---

## 11. Contribution to Thesis

This module contributes to Chapter 3 (Dataset Analysis) by validating dataset organization before preprocessing and model development.

It establishes the foundation for all subsequent stages of the segmentation pipeline.

---

## 12. Contribution to Future Paper

The Dataset Explorer provides a reusable dataset validation component that can be integrated into future multimodal medical imaging frameworks.

Its modular design improves reproducibility and reduces preprocessing errors.