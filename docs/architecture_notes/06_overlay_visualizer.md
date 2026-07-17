# OverlayVisualizer

The **OverlayVisualizer** is a visualization component responsible for rendering a two-dimensional medical image together with its corresponding segmentation mask. It overlays the segmentation mask on the original image using configurable transparency and independent colormaps, enabling qualitative assessment of segmentation accuracy.

The class is part of the visualization layer of the medical image segmentation framework and extends the **BaseVisualizer** abstract class. It is designed to provide consistent, reusable, and publication-quality visualizations while remaining completely independent of data loading and preprocessing logic.

---

# Table of Contents

- [OverlayVisualizer](#overlayvisualizer)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Purpose](#purpose)
- [Responsibilities](#responsibilities)
- [Non-Responsibilities](#non-responsibilities)
- [Design Principles](#design-principles)
  - [Single Responsibility Principle (SRP)](#single-responsibility-principle-srp)
  - [Reusability](#reusability)
  - [Stateless Design](#stateless-design)
  - [Extensibility](#extensibility)
  - [Consistency](#consistency)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Public API](#public-api)
  - [`render()`](#render)
    - [Parameters](#parameters)
    - [Returns](#returns)
    - [Raises](#raises)
      - [`TypeError`](#typeerror)
      - [`ValueError`](#valueerror)
- [Integration Within the Framework](#integration-within-the-framework)
- [Example Usage](#example-usage)
- [Testing and Validation](#testing-and-validation)
  - [Integration Tests](#integration-tests)
  - [Visual Validation](#visual-validation)
- [Advantages](#advantages)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [Key Features](#key-features)

---

# Overview

The **OverlayVisualizer** provides a reusable interface for visualizing a medical image together with its corresponding segmentation mask.

Rather than displaying the image and mask separately, the module overlays the segmentation mask on the original image using configurable transparency and independent colormaps. This enables rapid qualitative assessment of anatomical structures, segmentation boundaries, and annotation quality.

The visualizer accepts both the medical image and segmentation mask as **NumPy arrays** and returns a **Matplotlib Figure**, making it compatible with a wide range of medical image segmentation workflows.

---

# Purpose

Medical image segmentation is commonly evaluated using both **quantitative metrics** and **qualitative visual inspection**.

Although metrics such as the **Dice Similarity Coefficient (DSC)** and **Intersection over Union (IoU)** quantify segmentation performance, they do not reveal localized prediction errors, anatomical misalignment, or boundary inaccuracies.

The purpose of the **OverlayVisualizer** is to provide a reusable mechanism for displaying segmentation masks directly on top of their corresponding medical images. This allows researchers to:

- Inspect segmentation boundaries
- Verify annotation quality
- Identify failure cases
- Compare predicted regions with anatomical structures
- Improve qualitative model evaluation

---

# Responsibilities

The **OverlayVisualizer** is responsible for:

- Rendering a two-dimensional medical image
- Overlaying a segmentation mask on the image
- Applying configurable mask transparency
- Supporting independent colormaps for image and mask
- Supporting configurable interpolation methods
- Applying optional figure titles
- Configuring axis visibility
- Returning a Matplotlib `Figure` object for further processing

---

# Non-Responsibilities

To comply with the **Single Responsibility Principle (SRP)**, the **OverlayVisualizer** does **not** perform:

- Reading DICOM images
- Reading segmentation masks
- Image preprocessing
- Image normalization
- Image resizing
- Dataset management
- Saving rendered figures
- Comparison visualization
- Prediction visualization
- Interactive visualization

These responsibilities are delegated to dedicated framework components.

---

# Design Principles

The **OverlayVisualizer** follows modern software engineering principles.

## Single Responsibility Principle (SRP)

The module performs only one task:

> Render a medical image together with its segmentation mask.

---

## Reusability

The visualizer is completely independent of:

- Dataset
- Imaging modality
- Deep learning framework
- Segmentation architecture

It can therefore be reused across CT, MRI, PET, Ultrasound, and other medical imaging applications.

---

## Stateless Design

The class stores no internal state.

Every rendering operation depends solely on the supplied input arguments, ensuring predictable and reproducible behaviour.

---

## Extensibility

Future visualization capabilities such as:

- Contour overlays
- Confidence maps
- Legends
- Transparency controls
- Interactive viewers

can be added without modifying the existing implementation.

---

## Consistency

The **OverlayVisualizer** inherits reusable helper methods from **BaseVisualizer**, ensuring a consistent visualization workflow across all visualization modules.

---

# Architecture

The **OverlayVisualizer** extends the reusable **BaseVisualizer**.

```text
                    BaseVisualizer
                           │
                           ▼
                 OverlayVisualizer
                           │
                           ▼
                  Matplotlib Figure
```

The rendering process reuses helper methods provided by the base class for:

- Figure creation
- Title application
- Axis configuration
- Layout management

---

# Workflow

The visualization workflow consists of the following sequence.

```text
Medical Image (NumPy Array)
            │
            ▼
Segmentation Mask (NumPy Array)
            │
            ▼
Input Validation
            │
            ▼
Create Figure
            │
            ▼
Render Medical Image
            │
            ▼
Overlay Segmentation Mask
            │
            ▼
Apply Title
            │
            ▼
Configure Axes
            │
            ▼
Apply Layout
            │
            ▼
Return Figure
```

This workflow ensures that visualization is performed only after validating all user inputs, improving reliability and reducing runtime errors.

---

# Public API

## `render()`

Renders a two-dimensional medical image together with its corresponding segmentation mask.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `image` | Two-dimensional medical image represented as a NumPy array. |
| `mask` | Two-dimensional segmentation mask with the same dimensions as the image. |
| `mask_alpha` | Transparency of the segmentation mask. Must be between **0.0** and **1.0**. |
| `image_cmap` | Matplotlib colormap used to render the medical image. |
| `mask_cmap` | Matplotlib colormap used to render the segmentation mask. |
| `interpolation` | Interpolation method applied to both image and mask. |
| `title` | Optional figure title. |
| `figsize` | Figure dimensions (in inches). |
| `dpi` | Figure resolution (dots per inch). |
| `show_axes` | Controls whether axes are displayed. |

### Returns

Returns a **Matplotlib `Figure`** object containing the rendered medical image and segmentation mask overlay.

### Raises

#### `TypeError`

Raised when:

- `image` is not a NumPy array
- `mask` is not a NumPy array
- `mask_alpha` is not numeric

#### `ValueError`

Raised when:

- `image` is not two-dimensional
- `mask` is not two-dimensional
- Image and mask dimensions differ
- `mask_alpha` falls outside the valid range **[0.0, 1.0]**

---

# Integration Within the Framework

The **OverlayVisualizer** integrates seamlessly with multiple components of the medical image segmentation framework.

```text
                 CHAOS Dataset
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
   DICOMReader                GroundTruthReader
        │                             │
        ▼                             ▼
  Medical Image            Segmentation Mask
        │                             │
        └──────────────┬──────────────┘
                       ▼
               OverlayVisualizer
                       ▼
              Matplotlib Figure
```

The **OverlayVisualizer** serves as the primary visualization component for displaying segmentation results during:

- Dataset exploration
- Annotation verification
- Model evaluation
- Debugging
- Qualitative analysis
- Publication figure generation

Its modular design enables integration with any dataset reader or segmentation model without requiring modifications to the visualization layer.

---

# Example Usage

```python
from src.data.dicom_reader import DICOMReader
from src.data.ground_truth_reader import GroundTruthReader
from src.visualization.overlay_visualizer import OverlayVisualizer

image = DICOMReader().read_image(dicom_path)
mask = GroundTruthReader().read_mask(mask_path)

visualizer = OverlayVisualizer()

figure = visualizer.render(
    image=image,
    mask=mask,
    mask_alpha=0.4,
    title="CHAOS CT Liver Overlay"
)
```

The returned figure can subsequently be:

- Displayed interactively
- Saved to disk
- Embedded into reports or publications
- Included in experiment documentation
- Used for qualitative comparison of segmentation results

---

# Testing and Validation

The **OverlayVisualizer** has been validated using the real **CHAOS CT dataset**.

## Integration Tests

The validation process verifies:

- Successful loading of a real DICOM CT image
- Successful loading of the corresponding liver ground truth mask
- Matching image and mask dimensions
- Correct rendering of the segmentation overlay
- Correct return type (`matplotlib.figure.Figure`)
- Successful saving of the rendered figure
- Verification that the output file exists and is non-empty
- Proper release of Matplotlib resources after rendering

## Visual Validation

Qualitative inspection confirmed:

- Correct anatomical alignment of the liver mask
- Appropriate transparency blending
- Accurate overlay rendering
- No visible distortion or spatial misalignment

These validation results demonstrate that the visualizer performs reliably under real-world research conditions.

---

# Advantages

The **OverlayVisualizer** offers several advantages:

- Simple and reusable interface
- Dataset-independent implementation
- Publication-quality visualizations
- Configurable transparency and colormaps
- Consistent rendering workflow across the framework
- Defensive input validation
- Easy integration with different segmentation models
- Compatible with CT, MRI, PET, and other medical imaging modalities

---

# Current Limitations

The current implementation focuses on **single-image overlays**.

The following features are intentionally excluded from **Version 1**:

- Multi-class segmentation overlays
- Contour visualization
- Automatic color legends
- Confidence and uncertainty heatmaps
- Interactive visualization
- Comparison of overlays from multiple models
- Three-dimensional volume rendering
- Automatic figure export

These capabilities will be introduced in dedicated visualization modules to preserve the **Single Responsibility Principle (SRP)** and maintain a clean, modular architecture.

---

# Future Work

The visualization subsystem will continue to evolve with support for advanced qualitative evaluation features.

Planned improvements include:

- Multi-class mask visualization
- Contour-only overlays
- Confidence and uncertainty visualization
- Automatic legend generation
- Interactive visualization controls
- Adjustable overlay blending modes
- Support for prediction overlays
- Three-dimensional volume visualization
- Integration with future **ComparisonVisualizer**
- Integration with future **PredictionVisualizer**

These enhancements will expand the framework while preserving the reusable architecture established by the **BaseVisualizer**.

---

# Conclusion

The **OverlayVisualizer** provides a robust, reusable, and extensible solution for qualitative assessment of medical image segmentation.

By overlaying segmentation masks directly on medical images, it enables researchers to visually inspect anatomical alignment, verify annotation quality, identify segmentation errors, and assess model performance throughout the development lifecycle.

Its stateless architecture, strict input validation, reusable design, and seamless integration with the framework's visualization layer make it a dependable component for:

- Dataset exploration
- Annotation verification
- Model debugging
- Qualitative evaluation
- Publication-quality figure generation

Together with the **BaseVisualizer** and **MaskVisualizer**, the **OverlayVisualizer** forms a core component of the visualization subsystem within the medical image segmentation framework.

---

# Key Features

- Single Responsibility Principle (SRP) compliant
- Stateless architecture
- Framework-independent implementation
- Dataset-independent design
- Publication-quality visualizations
- Configurable transparency and colormaps
- Defensive input validation
- Compatible with CT, MRI, PET, and other modalities
- Seamless integration with the visualization framework
- Extensible foundation for future visualization modules