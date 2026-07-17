# MaskVisualizer

The **MaskVisualizer** is a visualization component responsible for rendering a single segmentation mask as a two-dimensional image. It converts a binary or labeled segmentation mask represented as a NumPy array into a Matplotlib figure for visual inspection, debugging, and qualitative evaluation.

The module is intentionally lightweight and follows the **Single Responsibility Principle (SRP)** by focusing exclusively on visualization. It does not perform image loading, mask preprocessing, prediction generation, or dataset management.

---

# Table of Contents

- [MaskVisualizer](#maskvisualizer)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Purpose](#purpose)
- [Responsibilities](#responsibilities)
- [Non-Responsibilities](#non-responsibilities)
- [Design Principles](#design-principles)
  - [Single Responsibility Principle (SRP)](#single-responsibility-principle-srp)
  - [Separation of Concerns](#separation-of-concerns)
  - [Reusability](#reusability)
  - [Framework Independence](#framework-independence)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Summary](#summary)
- [Public API](#public-api)
  - [`render()`](#render)
    - [Parameters](#parameters)
    - [Returns](#returns)
- [Integration Within the Framework](#integration-within-the-framework)
- [Example Usage](#example-usage)
- [Testing](#testing)
  - [Integration Tests](#integration-tests)
- [Advantages](#advantages)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [Key Features](#key-features)

---

# Overview

The **MaskVisualizer** provides a reusable interface for displaying segmentation masks during dataset verification, debugging, and qualitative evaluation.

Instead of coupling visualization with dataset loading or model inference, this module isolates the rendering process into a dedicated component. This modular design improves maintainability, encourages code reuse, and aligns with clean software engineering practices.

The visualizer accepts a segmentation mask as a **NumPy array** and returns a **Matplotlib Figure**, making it compatible with a wide range of medical imaging workflows.

---

# Purpose

Medical image segmentation workflows require frequent inspection of both ground truth annotations and model predictions.

Before training segmentation architectures such as:

- U-Net
- Attention U-Net
- TransUNet
- UNETR
- Swin UNETR

it is essential to verify that segmentation masks are loaded correctly and accurately represent the annotated structures.

The **MaskVisualizer** provides a reusable and framework-independent interface for rendering segmentation masks while remaining independent of any specific dataset, preprocessing pipeline, or deep learning model.

---

# Responsibilities

The **MaskVisualizer** is responsible for:

- Rendering a single segmentation mask
- Displaying binary masks
- Displaying multi-class segmentation masks
- Supporting any Matplotlib colormap
- Configuring figure size and resolution
- Applying an optional figure title
- Configuring axis visibility
- Returning a Matplotlib `Figure` object for further processing

---

# Non-Responsibilities

The following tasks are intentionally **outside the scope** of this module:

- Reading segmentation masks from disk
- Reading DICOM images
- Reading NIfTI images
- Dataset management
- Ground truth analysis
- Image preprocessing
- Overlay visualization
- Comparison of multiple masks
- Model prediction visualization
- Saving figures to disk
- Performance evaluation

These responsibilities are delegated to dedicated modules within the overall segmentation framework.

---

# Design Principles

The **MaskVisualizer** follows several core software engineering principles.

## Single Responsibility Principle (SRP)

The module performs only one task:

> Render a segmentation mask.

---

## Separation of Concerns

Visualization is isolated from data loading, preprocessing, and model inference.

This separation simplifies maintenance and testing while reducing module coupling.

---

## Reusability

The visualizer accepts a generic **NumPy array**, allowing it to operate with virtually any segmentation dataset or deep learning framework.

---

## Framework Independence

The module has **no dependency** on:

- PyTorch
- TensorFlow
- MONAI
- nnU-Net
- Any specific segmentation architecture

This ensures maximum portability across research projects.

---

# Architecture

The **MaskVisualizer** extends the reusable **BaseVisualizer**, inheriting common visualization utilities such as figure creation, title configuration, axis management, and layout handling.

```text
BaseVisualizer
       │
       ▼
MaskVisualizer
       │
       ▼
Matplotlib Figure
```

The module exposes a single public interface:

```text
render(mask) → Figure
```

---

# Workflow

The rendering pipeline is intentionally simple and modular.

```text
GroundTruthReader
        │
        ▼
NumPy Array (Mask)
        │
        ▼
MaskVisualizer.render()
        │
        ▼
Create Figure
        │
        ▼
Display Mask
        │
        ▼
Return Figure
```

This workflow keeps visualization completely independent from data acquisition, preprocessing, and model inference, promoting a clean and extensible architecture.

---

# Summary

The **MaskVisualizer** serves as the foundational visualization component within the medical image segmentation framework.

By focusing solely on rendering segmentation masks, it provides:

- Simple and reusable visualization
- Clean software architecture
- Framework independence
- Easy integration with future modules
- Support for qualitative evaluation and debugging

Its lightweight design makes it an essential building block for inspecting segmentation masks throughout the development and evaluation pipeline.

---

# Public API

## `render()`

Renders a segmentation mask as a two-dimensional image.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `mask` | Segmentation mask represented as a NumPy array. |
| `title` | Optional figure title. |
| `cmap` | Matplotlib colormap used for rendering. |
| `figsize` | Width and height of the figure (in inches). |
| `dpi` | Figure resolution (dots per inch). |
| `show_axes` | Controls whether figure axes are displayed. |

### Returns

Returns a **Matplotlib `Figure`** object containing the rendered segmentation mask.

---

# Integration Within the Framework

The **MaskVisualizer** operates within the visualization layer of the framework after segmentation masks have been loaded by the data layer.

```text
CHAOS Dataset
        │
        ▼
GroundTruthReader
        │
        ▼
Segmentation Mask
        │
        ▼
MaskVisualizer
        │
        ▼
Matplotlib Figure
```

This modular architecture allows the same visualization component to be reused across future datasets and segmentation models without modification.

---

# Example Usage

```python
from src.data.ground_truth_reader import GroundTruthReader
from src.visualization.mask_visualizer import MaskVisualizer

reader = GroundTruthReader()

mask = reader.read_mask(mask_path)

visualizer = MaskVisualizer()

figure = visualizer.render(
    mask,
    title="CHAOS Liver Ground Truth Mask"
)
```

The returned figure can subsequently be:

- Displayed interactively
- Saved to disk
- Embedded into reports or publications
- Included in experiment documentation

---

# Testing

The **MaskVisualizer** has been validated using the **CHAOS (Combined Healthy Abdominal Organ Segmentation)** dataset.

## Integration Tests

The validation process verifies:

- Successful loading of real ground truth segmentation masks
- Correct mask dimensions
- Boolean mask representation
- Label integrity
- Successful figure generation
- Successful figure export
- Compatibility with Matplotlib

Unlike synthetic testing, validation was performed using actual liver ground truth annotations from the CHAOS dataset to ensure reliable behaviour under real-world research conditions.

---

# Advantages

The current implementation provides several benefits:

- Simple and reusable interface
- Dataset-independent architecture
- Lightweight implementation
- Easy integration with future segmentation models
- Consistent visualization across the framework
- Minimal external dependencies
- Straightforward extensibility for future visualization modules

---

# Current Limitations

The current implementation intentionally supports **only a single segmentation mask**.

The following capabilities are planned as dedicated visualization modules:

- Image and mask overlay visualization
- Side-by-side comparison of multiple masks
- Ground truth versus prediction visualization
- Three-dimensional volume visualization
- Multi-class legend support
- Interactive visualization

Keeping these features separate preserves the **Single Responsibility Principle (SRP)** and maintains a clean, modular architecture.

---

# Future Work

The visualization subsystem will gradually evolve into a complete visualization framework.

```text
Visualization Framework

BaseVisualizer
      │
      ├── MaskVisualizer            ✓ Completed
      ├── OverlayVisualizer         □ Planned
      ├── ComparisonVisualizer      □ Planned
      ├── PredictionVisualizer      □ Planned
      └── VolumeVisualizer          □ Planned
```

Each visualization component will focus on a single responsibility while sharing a common architectural foundation provided by the **BaseVisualizer**.

This design promotes:

- Code reuse
- Consistent APIs
- Easier maintenance
- Improved scalability
- Framework extensibility

---

# Conclusion

The **MaskVisualizer** provides a clean, reusable, and extensible solution for rendering segmentation masks within the medical image segmentation framework.

By separating visualization from data loading and preprocessing, the module adheres to modern software engineering principles while supporting scalable and maintainable research software.

The component has been successfully integrated with the **GroundTruthReader** and validated using the real **CHAOS dataset**, establishing a reliable visualization foundation for future modules and deep learning segmentation architectures.

---

# Key Features

- Single Responsibility Principle (SRP) compliant
- Dataset-independent design
- Framework-independent implementation
- Matplotlib-based visualization
- Reusable across segmentation models
- Compatible with binary and multi-class masks
- Modular architecture
- Extensible visualization framework
- Research-ready implementation
- Fully integrated with the CHAOS segmentation pipeline