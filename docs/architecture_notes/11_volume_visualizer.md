# VolumeVisualizer

The **VolumeVisualizer** is a visualization component responsible for rendering and analyzing **three-dimensional (3D) medical image volumes** used in medical image segmentation research. It provides high-level visualization utilities for volumetric datasets such as **Computed Tomography (CT)** and **Magnetic Resonance Imaging (MRI)** while supporting qualitative evaluation of segmentation models.

Unlike two-dimensional visualization modules that operate on individual image slices, the **VolumeVisualizer** enables researchers to inspect reconstructed volumetric data from multiple anatomical perspectives. It supports visualization of anatomical slices, segmentation masks, prediction masks, overlay images, and multiplanar reconstructions, allowing comprehensive examination of both imaging data and segmentation results.

The module is part of the visualization subsystem and inherits from **BaseVisualizer**, ensuring a consistent interface and design philosophy throughout the framework.

---

# Table of Contents

- [Overview](#overview)
- [Purpose](#purpose)
- [Responsibilities](#responsibilities)
- [Non-Responsibilities](#non-responsibilities)
- [Features](#features)
- [Visualization Modes](#visualization-modes)
- [Supported Anatomical Planes](#supported-anatomical-planes)

---

# Overview

Medical imaging modalities such as CT and MRI naturally produce volumetric datasets composed of hundreds of sequential image slices. While individual slices provide valuable information, many anatomical structures can only be fully understood when examined in three dimensions.

The **VolumeVisualizer** addresses this requirement by providing standardized visualization utilities for reconstructed 3D medical image volumes.

It enables researchers to:

- Inspect volumetric anatomy
- Visualize segmentation masks
- Analyze prediction results
- Compare anatomical planes
- Generate publication-quality volumetric figures

The module operates independently of the segmentation architecture and therefore supports CNN-based, Transformer-based, and future hybrid segmentation models.

---

# Purpose

Medical image segmentation research increasingly relies on volumetric datasets where anatomical structures extend across multiple image slices.

The purpose of the **VolumeVisualizer** is to provide reusable visualization tools for qualitative analysis of reconstructed medical image volumes.

The module supports:

- Volume inspection
- Segmentation verification
- Prediction analysis
- Anatomical localization
- Three-dimensional interpretation
- Publication-quality visualization

It plays an important role during dataset exploration, model validation, benchmarking, and research publication.

---

# Responsibilities

The **VolumeVisualizer** is responsible for:

- Visualizing reconstructed 3D medical image volumes
- Displaying axial, coronal, and sagittal anatomical planes
- Rendering volumetric segmentation masks
- Displaying prediction masks for qualitative evaluation
- Overlaying segmentation predictions on original images
- Producing multiplanar reconstructions
- Exporting publication-quality figures
- Validating visualization inputs before rendering

---

# Non-Responsibilities

To comply with the **Single Responsibility Principle (SRP)**, the **VolumeVisualizer** does **not** perform:

- Reading DICOM files
- Reading segmentation masks
- Reconstructing 3D volumes from image slices
- Image preprocessing
- Image normalization
- Neural network inference
- Model training
- Quantitative metric computation
- Statistical analysis

These responsibilities are delegated to dedicated components within the research framework.

---

# Features

The **VolumeVisualizer** provides the following capabilities:

- Single-slice visualization
- Ground truth mask visualization
- Prediction visualization
- Overlay visualization
- Multiplanar reconstruction
- High-resolution figure export
- Configurable figure size
- Configurable DPI
- Automatic input validation
- Publication-quality rendering

---

# Visualization Modes

The module supports several visualization modes for volumetric medical imaging.

---

## 1. Slice Visualization

Displays a single anatomical slice extracted from a reconstructed medical image volume.

### Purpose

- Inspect reconstructed CT or MRI volumes
- Verify anatomical structures
- Validate slice extraction
- Explore volumetric datasets

### Supported Planes

- Axial
- Coronal
- Sagittal

### Output

```text
+----------------+
| Original Slice |
+----------------+
```

---

## 2. Ground Truth Visualization

Displays the original image slice together with its corresponding expert-annotated segmentation mask.

### Purpose

- Verify annotations
- Inspect segmentation labels
- Validate dataset quality
- Confirm anatomical correspondence

### Output

```text
+----------------+----------------------+
| Original Slice | Ground Truth Mask    |
+----------------+----------------------+
```

---

## 3. Prediction Visualization

Displays the original image together with a predicted segmentation mask.

### Purpose

- Qualitative model evaluation
- Prediction inspection
- Error analysis
- Segmentation verification

### Output

```text
+----------------+----------------+
| Original Slice | Prediction     |
+----------------+----------------+
```

---

## 4. Overlay Visualization

Superimposes the predicted segmentation mask on the original medical image.

### Purpose

- Anatomical localization
- Boundary verification
- Visual assessment of segmentation accuracy
- Clinical interpretation

### Output

```text
+----------------+----------------------+
| Original Slice | Prediction Overlay   |
+----------------+----------------------+
```

Overlay transparency can be adjusted using the configurable **alpha** parameter.

---

## 5. Multiplanar Visualization

Displays three orthogonal anatomical planes simultaneously.

### Views

- Axial
- Coronal
- Sagittal

### Purpose

- Full volumetric inspection
- Anatomical orientation
- Three-dimensional structure analysis
- Clinical interpretation

### Output

```text
+-----------+-----------+-----------+
|  Axial    | Coronal   | Sagittal  |
+-----------+-----------+-----------+
```

Multiplanar visualization enables comprehensive exploration of anatomical structures from complementary viewpoints.

---

# Supported Anatomical Planes

The **VolumeVisualizer** supports visualization using the three standard anatomical planes employed in medical imaging.

---

## Axial Plane

The **axial plane** consists of horizontal slices through the patient's body and is the most commonly used plane in CT imaging.

### Typical Applications

- Liver segmentation
- Brain segmentation
- Abdominal CT analysis
- Organ boundary inspection
- Slice-by-slice evaluation

---

## Coronal Plane

The **coronal plane** provides a front-to-back view of anatomical structures.

### Typical Applications

- Organ localization
- Shape verification
- Vertical anatomical inspection
- Evaluation of anatomical symmetry
- Assessment of superior-inferior relationships

---

## Sagittal Plane

The **sagittal plane** divides the body into left and right sections, providing a side-view representation of anatomy.

### Typical Applications

- Side-view anatomical inspection
- Evaluation of organ depth
- Spatial relationship analysis
- Assessment of longitudinal anatomical structures
- Three-dimensional anatomical interpretation

The combined use of axial, coronal, and sagittal views enables comprehensive visualization of volumetric medical datasets and supports accurate qualitative analysis of segmentation results.

---

## Sagittal Plane

The **sagittal plane** provides a side-view representation of anatomical structures by dividing the body into left and right sections. This perspective is particularly useful for examining the depth and longitudinal continuity of organs within reconstructed medical image volumes.

### Typical Applications

- Organ depth inspection
- Anatomical continuity analysis
- Three-dimensional anatomical assessment
- Spatial relationship evaluation
- Volumetric structure verification

---

# Public API

The **VolumeVisualizer** exposes a collection of high-level visualization methods for qualitative analysis of reconstructed three-dimensional medical image volumes.

---

## `render()`

Primary visualization method inherited from **BaseVisualizer**.

### Purpose

Provides the default visualization interface by rendering a single axial slice from the reconstructed volume.

### Returns

- `matplotlib.figure.Figure`

---

## `render_slice()`

Renders a single slice extracted from a reconstructed three-dimensional medical image volume.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `volume` | Three-dimensional medical image volume |
| `slice_index` | Index of the slice to visualize |
| `plane` | Anatomical viewing plane |

### Returns

- `matplotlib.figure.Figure`

---

## `render_mask()`

Displays an original image slice together with its corresponding ground truth segmentation mask.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `image_volume` | Original reconstructed volume |
| `mask_volume` | Ground truth segmentation volume |
| `slice_index` | Slice index |
| `plane` | Anatomical viewing plane |

### Returns

- `matplotlib.figure.Figure`

---

## `render_prediction()`

Displays an original image slice together with the predicted segmentation mask.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `image_volume` | Original reconstructed volume |
| `prediction_volume` | Predicted segmentation volume |
| `slice_index` | Slice index |
| `plane` | Anatomical viewing plane |

### Returns

- `matplotlib.figure.Figure`

---

## `render_overlay()`

Displays a segmentation prediction overlaid on the corresponding medical image.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `image_volume` | Original reconstructed volume |
| `prediction_volume` | Predicted segmentation volume |
| `slice_index` | Slice index |
| `plane` | Anatomical viewing plane |
| `alpha` | Overlay transparency |

### Returns

- `matplotlib.figure.Figure`

---

## `render_multiplanar()`

Displays three orthogonal anatomical planes simultaneously.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `image_volume` | Three-dimensional medical image volume |
| `slice_indices` | Optional slice indices for each anatomical plane |

### Returns

- `matplotlib.figure.Figure`

---

## `save()`

Exports generated visualizations to disk.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `figure` | Matplotlib figure object |
| `output_path` | Destination file path |

### Behavior

The method automatically:

- Creates missing output directories
- Saves high-resolution publication-quality figures
- Closes the figure after saving to prevent memory leaks

---

# Input Validation

Before rendering any visualization, the **VolumeVisualizer** performs comprehensive validation to ensure input correctness and prevent undefined behavior.

The module validates:

- Volume is a NumPy array
- Volume is three-dimensional
- Mask and prediction volumes match the image volume dimensions
- Slice indices are valid integers
- Anatomical plane is a valid `VolumePlane`
- Overlay transparency (`alpha`) is numeric and lies within the interval **[0.0, 1.0]**
- Figure size values are positive
- DPI values are positive
- Output path is a valid string
- Figure object is a valid Matplotlib figure

Whenever invalid inputs are detected, descriptive exceptions are raised to simplify debugging and improve framework robustness.

---

# Integration within the Framework

The **VolumeVisualizer** integrates seamlessly with other components of the medical image segmentation framework.

### Dependent Modules

- BaseVisualizer
- VolumePlane
- DICOMReader
- GroundTruthReader

---

## Typical Workflow

```text
            DICOMReader
                 │
                 ▼
     Reconstructed 3D Volume
                 │
                 ▼
        GroundTruthReader
                 │
                 ▼
         VolumeVisualizer
                 │
      ┌──────────┼────────────┐
      │          │            │
      ▼          ▼            ▼
 Slice View   Mask View   Prediction View
      │          │            │
      └──────────┼────────────┐
                 ▼
         Overlay Visualization
                 │
                 ▼
     Multiplanar Visualization
```

This workflow illustrates how reconstructed medical image volumes are processed for qualitative visualization without modifying the underlying data.

---

# Usage Example

```python
from src.visualization.volume_visualizer import VolumeVisualizer

visualizer = VolumeVisualizer()

figure = visualizer.render_slice(
    volume=image_volume,
    slice_index=40,
)

figure = visualizer.render_mask(
    image_volume,
    mask_volume,
    slice_index=40,
)

figure = visualizer.render_prediction(
    image_volume,
    prediction_volume,
    slice_index=40,
)

figure = visualizer.render_overlay(
    image_volume,
    prediction_volume,
    slice_index=40,
    alpha=0.5,
)

figure = visualizer.render_multiplanar(
    image_volume,
)

visualizer.save(
    figure,
    "results/volume_visualization.png",
)
```

---

# Testing

The **VolumeVisualizer** has been comprehensively validated using both unit testing and integration testing to ensure correctness, robustness, and compatibility with the overall research framework.

---

## Unit Testing

The unit test suite verifies:

- Constructor behavior
- Slice rendering
- Ground truth visualization
- Prediction visualization
- Overlay visualization
- Multiplanar visualization
- Figure export
- Input validation
- Anatomical plane handling
- Exception handling
- Rendering consistency

Deterministic synthetic three-dimensional volumes are used throughout unit testing to ensure reproducibility and platform-independent execution.

---

## Integration Testing

Integration tests validate the complete visualization pipeline using the real **CHAOS (Combined Healthy Abdominal Organ Segmentation)** dataset.

The integration pipeline verifies:

- Reconstruction of three-dimensional image volumes from DICOM slices
- Alignment between CT volumes and liver segmentation masks
- Slice visualization
- Ground truth visualization
- Prediction visualization
- Overlay visualization
- Multiplanar visualization
- Publication-quality figure export
- End-to-end compatibility with the visualization framework

These tests demonstrate that the module functions correctly using real clinical imaging data while integrating seamlessly with the framework's data-loading components.

---

# Design Principles

The implementation follows established software engineering principles to ensure long-term maintainability and extensibility.

- **Single Responsibility Principle (SRP)** — Responsible exclusively for volumetric visualization.
- **Separation of Concerns** — Visualization logic is isolated from data loading, preprocessing, inference, and evaluation.
- **High Cohesion** — All methods contribute directly to volumetric visualization.
- **Low Coupling** — Interacts with other framework components through well-defined interfaces.
- **Reusability** — Compatible with any segmentation architecture operating on volumetric data.
- **Extensibility** — New visualization techniques can be added without modifying existing functionality.
- **Maintainability** — Modular helper methods reduce code duplication and simplify future enhancements.
- **Deterministic Testing** — Predictable behavior enables reproducible scientific experiments.
- **Publication-Quality Visualization** — Generates high-resolution figures suitable for theses, journals, and conference publications.

---

# Advantages

The **VolumeVisualizer** offers several advantages within the medical image segmentation research framework.

- Architecture-independent implementation
- Support for three-dimensional medical image volumes
- Multi-planar anatomical visualization
- Publication-quality figure generation
- Comprehensive input validation
- Robust exception handling
- Modular and reusable design
- Seamless integration with the visualization subsystem
- Consistent visualization API inherited from **BaseVisualizer**
- Easily extensible for future volumetric visualization techniques

---

# Current Limitations

The current implementation focuses on qualitative visualization of reconstructed volumetric datasets.

The following capabilities are intentionally excluded from the present version and are reserved for future development:

- Interactive three-dimensional rendering
- GPU-accelerated volume rendering
- Confidence and uncertainty visualization
- Multi-class volumetric overlays
- Temporal visualization for longitudinal imaging studies
- Animated slice navigation
- Virtual reality (VR) and augmented reality (AR) visualization support

Restricting the scope of the current implementation ensures a robust and maintainable foundation while allowing future enhancements without affecting existing functionality.

---

# Future Extensions

Future versions of the **VolumeVisualizer** may include:

## Interactive Volume Exploration

Interactive slice navigation using libraries such as:

- Napari
- Plotly
- VTK

---

## True 3D Volume Rendering

Support full volumetric rendering of CT and MRI datasets rather than slice-based visualization.

---

## Multi-Class Volume Visualization

Display multiple anatomical structures simultaneously using configurable color maps.

---

## Confidence and Uncertainty Maps

Visualize voxel-level prediction confidence generated by deep learning models.

---

## Boundary Error Visualization

Highlight volumetric boundary discrepancies between predictions and ground truth annotations.

---

## Animated Navigation

Generate animated traversals through volumetric datasets for presentation and educational purposes.

---

## Automatic Report Generation

Produce standardized volumetric visualization reports suitable for:

- Thesis chapters
- Research publications
- Benchmark reports
- Clinical demonstrations

---

# Module Summary

The **VolumeVisualizer** provides a comprehensive solution for qualitative visualization of three-dimensional medical image segmentation data.

By supporting visualization across multiple anatomical planes, inspection of ground truth and predicted segmentation masks, overlay rendering, and multiplanar reconstruction, the module enables detailed exploration of volumetric CT and MRI datasets throughout the research lifecycle.

Its modular architecture, comprehensive validation, publication-quality output, and seamless integration with the broader medical image segmentation framework make it a valuable component for dataset exploration, model development, benchmarking, qualitative analysis, and scientific publication.

Together with the **BaseVisualizer**, **MaskVisualizer**, **OverlayVisualizer**, **ComparisonVisualizer**, and **PredictionVisualizer**, the **VolumeVisualizer** completes the visualization subsystem, providing a unified and extensible platform for reproducible, evidence-based research in multimodal medical image segmentation.