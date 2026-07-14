# GroundTruthReader

## Overview

The **GroundTruthReader** is responsible for loading, validating, and analyzing segmentation masks from the **CHAOS (Combined Healthy Abdominal Organ Segmentation)** dataset. It provides a unified interface for reading binary ground truth masks, extracting statistical information, identifying class labels, and validating annotations before they are used in preprocessing or deep learning pipelines.

Rather than simply reading image files, this module ensures that segmentation masks are correctly interpreted, validated, and transformed into a format suitable for downstream processing. It serves as one of the fundamental building blocks of the medical image analysis framework.

---

# Why This Module Exists

Deep learning models do not learn directly from medical images alone. Instead, they learn from **image–mask pairs**, where:

- The medical image contains anatomical information.
- The segmentation mask contains expert annotations identifying the target organ.

Without accurate ground truth annotations, supervised segmentation models such as U-Net, Attention U-Net, TransUNet, and UNETR cannot learn meaningful representations.

Therefore, a reliable mechanism for reading, validating, and analyzing segmentation masks is essential before preprocessing, dataset loading, or model training begins.

The GroundTruthReader addresses this requirement by providing a clean and extensible interface for handling segmentation masks while ensuring consistency throughout the data pipeline.

---

# Medical Imaging Background

## What is a Segmentation Mask?

A segmentation mask is an image where each pixel represents a semantic class instead of an intensity value.

For binary liver segmentation:

- Background pixels → 0
- Liver pixels → 255

Unlike CT or MRI images, segmentation masks do not contain anatomical texture or grayscale information. Instead, they indicate the precise location of the target organ.

---

## Why is Ground Truth Important?

Ground truth annotations act as the reference standard used during supervised learning.

During training, the model compares its predicted segmentation against the ground truth mask and minimizes the difference using loss functions such as:

- Dice Loss
- Binary Cross Entropy
- Focal Loss
- Tversky Loss

Without reliable ground truth masks, segmentation performance cannot be accurately optimized or evaluated.

---

## Who Creates Ground Truth Masks?

Ground truth masks are manually annotated by experienced radiologists or medical imaging experts.

These annotations undergo rigorous quality assurance to ensure that organ boundaries are accurately represented.

---

## Why Binary Masks?

The CHAOS CT dataset focuses exclusively on liver segmentation.

Consequently, each pixel belongs to one of only two classes:

| Pixel Value | Meaning |
|-------------|----------|
| 0 | Background |
| 255 | Liver |

Binary masks simplify the segmentation task and reduce computational complexity during training.

---

## Difference Between Image and Mask

| Medical Image | Segmentation Mask |
|---------------|-------------------|
| Contains grayscale intensity values | Contains class labels |
| Represents anatomical structures | Represents expert annotations |
| Used as model input | Used as training target |

---

# CHAOS Dataset Structure

```text
CHAOS/

└── CT/
    └── Patient/
        ├── DICOM_anon/
        │   ├── image001.dcm
        │   ├── image002.dcm
        │   └── ...
        │
        └── Ground/
            ├── liver_GT_001.png
            ├── liver_GT_002.png
            └── ...

Each CT slice has a corresponding segmentation mask.

The filenames follow a consistent ordering, enabling direct pairing between images and masks during dataset loading.

This one-to-one correspondence is critical for supervised segmentation.

---

# Module Architecture

```text
GroundTruthReader

├── read_mask()
├── unique_labels()
├── mask_statistics()
└── is_empty_mask()
```

Each method has a single responsibility:

| Method | Responsibility |
|---------|----------------|
| `read_mask()` | Load PNG mask |
| `unique_labels()` | Identify class labels |
| `mask_statistics()` | Compute mask statistics |
| `is_empty_mask()` | Detect missing annotations |

Visualization functionality is intentionally implemented in a separate module to maintain separation of concerns and adhere to the **Single Responsibility Principle (SRP)**.

---

# Public API

## `read_mask()`

### Purpose

Loads a segmentation mask from disk and converts it into a NumPy array.

### Input

```python
path: Path
```

### Output

```python
numpy.ndarray
```

### Raises

- `FileNotFoundError`
- `ValueError`

### Example

```python
mask = reader.read_mask(mask_path)
```

---

## `unique_labels()`

### Purpose

Returns all unique pixel values present in the segmentation mask.

### Input

```python
mask: np.ndarray
```

### Output

```python
numpy.ndarray
```

### Example

```python
labels = reader.unique_labels(mask)
```

---

## `mask_statistics()`

### Purpose

Computes useful statistics describing the segmentation mask.

### Output

A `MaskStatistics` dataclass containing:

- Total pixels
- Foreground pixels
- Background pixels
- Foreground ratio
- Unique labels

---

## `is_empty_mask()`

### Purpose

Determines whether a segmentation mask contains any annotated foreground pixels.

### Returns

```python
bool
```

### Example

```python
True
```

Returns `True` if the mask contains only background pixels.

---

# Method-by-Method Explanation

## `read_mask()`

This function acts as the entry point for loading segmentation masks.

It validates file existence, ensures correct image loading, and converts the image into a NumPy array suitable for further processing.

---

## `unique_labels()`

Medical datasets may contain:

- Binary masks
- Multi-class masks
- Instance masks

Using `np.unique()` automatically detects all class labels without hardcoding pixel values.

Returning a NumPy array preserves compatibility with scientific computing libraries.

---

## `mask_statistics()`

Rather than repeatedly calculating statistics throughout the project, this function centralizes all mask analysis in one reusable location.

The resulting dataclass provides structured information for dataset validation and quality assurance.

---

## `is_empty_mask()`

Some datasets may contain slices without annotated organs.

Instead of counting every pixel manually, the implementation uses `np.any()` for efficient foreground detection.

This approach offers:

- Fast execution
- Readable implementation
- Constant memory usage

---

# Data Flow

```text
PNG Mask
     │
     ▼
GroundTruthReader.read_mask()
     │
     ▼
NumPy Array
     │
     ▼
mask_statistics()
     │
     ▼
MaskStatistics
     │
     ▼
Dataset Loader
     │
     ▼
Preprocessing
     │
     ▼
U-Net Training
```

This workflow illustrates how segmentation masks move through the overall medical image analysis pipeline.

---

# Design Decisions

Several engineering decisions were made to improve maintainability and extensibility.

## Why Dataclasses?

Dataclasses provide a clean and structured representation of statistical results while reducing boilerplate code.

---

## Why Separate Visualization?

Visualization is intentionally isolated from data access.

### Benefits

- Better modularity
- Easier maintenance
- Improved testability
- Compliance with the Single Responsibility Principle (SRP)

---

## Why `np.any()`?

`np.any()` quickly determines whether any foreground pixels exist without explicitly counting all pixels.

---

## Why Preserve NumPy Data Types?

Maintaining original NumPy data types avoids unnecessary memory allocations and ensures compatibility with downstream image processing libraries.

---

## Why Use Keyword-Only Arguments (Future)?

Keyword-only parameters improve readability and reduce ambiguity as APIs become more complex.

---

# Testing

The module was validated using segmentation masks from the CHAOS dataset.

## Test Case

### Input

```text
liver_GT_001.png
```

### Expected Result

- Mask loads successfully.
- Unique labels detected.
- Statistics computed.
- Mask correctly classified as non-empty.

### Console Output

```text
============================================================
EMPTY MASK TEST
============================================================
Is Empty Mask : False
============================================================
```

The observed output matches the expected behavior, confirming the correct implementation of the module.

---

# Future Extensions

The current implementation is designed to support future research developments.

Possible extensions include:

- Multi-class segmentation masks
- NIfTI (`.nii` / `.nii.gz`) support
- 3D volumetric segmentation
- Instance segmentation
- Overlay visualization
- Automatic quality assurance reports
- Dataset-wide statistical analysis

---

# Connection to Thesis

The **GroundTruthReader** occupies an essential position within the overall research framework.

```text
Dataset Explorer
        │
        ▼
DICOM Reader
        │
        ▼
Metadata Scanner
        │
        ▼
GroundTruthReader
        │
        ▼
Dataset Loader
        │
        ▼
Preprocessing
        │
        ▼
U-Net
        │
        ▼
Attention U-Net
        │
        ▼
TransUNet
        │
        ▼
UNETR
        │
        ▼
Swin UNETR
```

This modular pipeline promotes reusability, maintainability, and scalability throughout the research project.

---

# Key Terminology

| Term | Definition |
|------|------------|
| **Ground Truth** | Expert annotation used as the reference during supervised learning |
| **Segmentation Mask** | Image containing semantic class labels instead of intensity values |
| **Foreground Pixel** | Pixel representing the target organ |
| **Background Pixel** | Pixel representing non-target regions |
| **Class Label** | Integer value assigned to a semantic category |
| **Binary Segmentation** | Segmentation involving only two classes |
| **Class Imbalance** | Unequal distribution of foreground and background pixels |
| **Predicate Function** | Function returning a Boolean value |
| **Dataclass** | Python structure for organizing related data |
| **API Contract** | Expected behavior of a public interface |
| **Data Contract** | Agreement regarding data format and structure |
| **Type Fidelity** | Preservation of original data types |
| **Short-Circuit Evaluation** | Early termination of logical evaluation when possible |

---

# Summary

The **GroundTruthReader** provides a modular and extensible interface for reading and validating segmentation masks within the CHAOS dataset.

By separating data access, analysis, and visualization into independent components, the framework adheres to the **Single Responsibility Principle (SRP)** while establishing a robust foundation for preprocessing, dataset loading, quality assurance, and deep learning model development.

Its modular architecture enables seamless integration with downstream pipelines, including preprocessing, dataset management, and state-of-the-art segmentation networks such as **U-Net**, **Attention U-Net**, **TransUNet**, **UNETR**, and **Swin UNETR**, making it a reliable and scalable component of the overall medical image analysis framework.