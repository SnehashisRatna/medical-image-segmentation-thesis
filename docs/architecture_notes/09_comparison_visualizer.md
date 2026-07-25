# ComparisonVisualizer

The **ComparisonVisualizer** is a visualization component responsible for generating qualitative comparisons between segmentation results produced by different deep learning models. It provides a unified interface for displaying the original medical image, the corresponding ground truth segmentation mask, multiple predicted segmentation masks, overlay visualizations, and pixel-wise difference maps.

Unlike quantitative evaluation metrics such as the **Dice Similarity Coefficient (DSC)**, **Intersection over Union (IoU)**, or **Hausdorff Distance (HD95)**, this module focuses entirely on **qualitative analysis**. Visual inspection enables researchers to identify prediction boundaries, missed regions, false positives, false negatives, anatomical inconsistencies, and structural differences that numerical metrics alone cannot fully describe.

The **ComparisonVisualizer** is designed as a reusable visualization component within the medical image segmentation framework. It is independent of any specific segmentation architecture and can therefore be used with both convolutional neural network (CNN) and transformer-based segmentation models.

---

# Table of Contents

- [ComparisonVisualizer](#comparisonvisualizer)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Purpose](#purpose)
- [Responsibilities](#responsibilities)
- [Non-Responsibilities](#non-responsibilities)
- [Design Principles](#design-principles)
  - [Single Responsibility Principle (SRP)](#single-responsibility-principle-srp)
  - [Open/Closed Principle (OCP)](#openclosed-principle-ocp)
  - [Reusability](#reusability)
  - [Maintainability](#maintainability)
- [Architecture](#architecture)
- [Public Methods](#public-methods)
  - [`render()`](#render)
  - [`compare_predictions()`](#compare_predictions)
- [Public Methods](#public-methods-1)
  - [`render()`](#render-1)
  - [`compare_predictions()`](#compare_predictions-1)
  - [`compare_models()`](#compare_models)
  - [`compare_overlays()`](#compare_overlays)
  - [`compare_difference()`](#compare_difference)
  - [`save()`](#save)
- [Internal Helper Methods](#internal-helper-methods)
  - [`_validate_inputs()`](#_validate_inputs)
  - [`_validate_masks()`](#_validate_masks)
  - [`_validate_alpha()`](#_validate_alpha)
  - [`_validate_dpi()`](#_validate_dpi)
  - [`_create_comparison_figure()`](#_create_comparison_figure)
  - [`_display_array()`](#_display_array)
  - [`_display_mask()`](#_display_mask)
  - [`_display_overlay()`](#_display_overlay)
  - [`_copy_images()`](#_copy_images)
- [Input Validation](#input-validation)
- [Output](#output)
- [Current Workflow](#current-workflow)
- [Future Integration](#future-integration)
- [Supported Segmentation Models](#supported-segmentation-models)
- [Future Extensions](#future-extensions)
  - [Multi-Class Segmentation](#multi-class-segmentation)
  - [Contour Visualization](#contour-visualization)
  - [Error Visualization](#error-visualization)
  - [Interactive Visualization](#interactive-visualization)
  - [3D Volume Visualization](#3d-volume-visualization)
  - [Confidence Visualization](#confidence-visualization)
  - [Automatic Report Generation](#automatic-report-generation)
- [Testing Strategy](#testing-strategy)
  - [Unit Testing](#unit-testing)
  - [Integration Testing](#integration-testing)
- [Advantages](#advantages)
- [Conclusion](#conclusion)

---

# Overview

The **ComparisonVisualizer** provides a standardized framework for visually comparing segmentation results produced by multiple deep learning architectures.

Rather than generating individual visualization scripts for each segmentation model, this module offers a single reusable interface capable of displaying outputs from any segmentation network.

By standardizing qualitative visualization, the module promotes:

- Reproducibility
- Consistency
- Maintainability
- Fair comparison across architectures

It plays a critical role during benchmarking, error analysis, and research gap identification.

---

# Purpose

Quantitative evaluation metrics provide an overall measure of segmentation performance but rarely explain **why** one model performs better than another.

The purpose of the **ComparisonVisualizer** is to bridge this gap by enabling researchers to inspect segmentation results visually.

Using a unified visualization pipeline, researchers can:

- Compare segmentation quality across multiple models
- Inspect anatomical boundaries
- Identify false positives and false negatives
- Detect over-segmentation and under-segmentation
- Analyze model failure cases
- Generate publication-quality comparison figures

Instead of creating custom visualization scripts for every experiment, this module provides a reusable implementation that supports any segmentation architecture.

---

# Responsibilities

The **ComparisonVisualizer** is responsible for:

- Displaying the original medical image
- Displaying the corresponding ground truth segmentation mask
- Displaying predictions from one or more segmentation models
- Creating overlay visualizations
- Generating pixel-wise difference maps
- Saving publication-quality comparison figures
- Validating images, masks, predictions, transparency values, and output resolution
- Producing reusable qualitative comparison figures for research experiments

---

# Non-Responsibilities

To maintain compliance with the **Single Responsibility Principle (SRP)**, the **ComparisonVisualizer** does **not** perform:

- Reading medical images from disk
- Reading segmentation masks
- Dataset management
- Image preprocessing
- Image normalization
- Neural network inference
- Model training
- Metric computation
- Statistical analysis
- Experiment management

These responsibilities belong to dedicated modules within the framework.

---

# Design Principles

The implementation follows modern software engineering principles.

## Single Responsibility Principle (SRP)

The module performs only one task:

> Generate qualitative comparisons between segmentation results.

It does **not**:

- Load datasets
- Perform preprocessing
- Execute neural network inference
- Compute evaluation metrics
- Train segmentation models

---

## Open/Closed Principle (OCP)

The module is open for extension but closed for modification.

Future visualization methods can be added without changing the existing implementation.

Examples include:

- Contour visualization
- Uncertainty visualization
- Attention map visualization
- Feature map visualization
- Confidence heatmaps

---

## Reusability

The visualizer accepts generic **NumPy arrays**, making it compatible with virtually any segmentation architecture regardless of its implementation.

---

## Maintainability

Visualization logic is divided into small helper methods.

This design:

- Reduces duplicated code
- Simplifies maintenance
- Improves readability
- Facilitates future extensions

---

# Architecture

```text
                Medical Image
                      │
                      ▼
             Ground Truth Mask
                      │
                      ▼
          Segmentation Predictions
                      │
                      ▼
           ComparisonVisualizer
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
Prediction     Overlay Comparison  Difference Map
Comparison
        │
        ▼
Publication-Quality Figure
```

The **ComparisonVisualizer** acts as the central qualitative analysis component of the visualization subsystem.

---

# Public Methods

## `render()`

Creates the default qualitative comparison consisting of:

- Original medical image
- Ground truth segmentation mask
- Predicted segmentation masks

Internally, this method delegates the visualization process to `compare_predictions()`.

---

## `compare_predictions()`

Displays:

- Original medical image
- Ground truth segmentation
- Predictions from multiple segmentation models

This is the primary interface used for side-by-side comparison of segmentation architectures and serves as the default visualization for qualitative benchmarking.

---

# Public Methods

## `render()`

Creates the default qualitative comparison consisting of:

- Original medical image
- Ground truth segmentation mask
- Predicted segmentation masks

Internally, this method delegates the visualization process to `compare_predictions()`.

---

## `compare_predictions()`

Displays:

- Original medical image
- Ground truth segmentation mask
- Predictions from multiple segmentation models

This is the primary interface for side-by-side comparison of segmentation architectures and serves as the default visualization for qualitative benchmarking.

---

## `compare_models()`

Provides an alias for `compare_predictions()`.

This method improves readability during experimental evaluation where the primary objective is comparing the outputs of multiple segmentation architectures.

---

## `compare_overlays()`

Creates overlay visualizations by superimposing segmentation masks on the corresponding medical image.

This visualization is particularly useful for evaluating:

- Boundary alignment
- Organ localization
- Segmentation coverage
- Missed anatomical regions
- Over-segmentation and under-segmentation

---

## `compare_difference()`

Generates pixel-wise difference maps between each prediction and the corresponding ground truth segmentation mask.

The current implementation highlights disagreement regions between prediction and annotation, providing a rapid visual indication of model errors.

Difference visualization assists in identifying:

- False positives
- False negatives
- Boundary inconsistencies
- Localization errors

---

## `save()`

Exports comparison figures to disk.

The method automatically creates missing output directories before saving high-resolution publication-quality figures.

---

# Internal Helper Methods

The implementation is organized into several reusable helper methods.

## `_validate_inputs()`

Validates:

- Medical image
- Ground truth mask
- Prediction masks

Ensures that all required arrays are present and have compatible dimensions before visualization begins.

---

## `_validate_masks()`

Validates:

- Prediction availability
- Prediction data types
- Mask dimensions
- Shape consistency

---

## `_validate_alpha()`

Ensures that overlay transparency values lie within the valid interval **[0.0, 1.0]**.

---

## `_validate_dpi()`

Validates the requested figure resolution before exporting publication-quality figures.

---

## `_create_comparison_figure()`

Creates a Matplotlib figure whose width automatically scales according to the number of comparison panels being displayed.

This allows the visualization layout to remain readable regardless of the number of segmentation models.

---

## `_display_array()`

Displays a generic NumPy array using configurable visualization parameters.

---

## `_display_mask()`

Uses the reusable **MaskVisualizer** component to render segmentation masks.

This avoids code duplication while maintaining consistency across the visualization framework.

---

## `_display_overlay()`

Uses the reusable **OverlayVisualizer** component to render image-mask overlays.

This ensures consistent overlay rendering throughout the framework.

---

## `_copy_images()`

Transfers rendered image layers between Matplotlib axes.

This helper enables the module to reuse existing visualization components while maintaining a clean, modular architecture.

---

# Input Validation

The **ComparisonVisualizer** performs comprehensive validation before generating visualizations.

The module validates:

- Missing medical images
- Missing ground truth masks
- Empty prediction dictionaries
- Mismatched image dimensions
- Invalid prediction data types
- Invalid overlay transparency values
- Invalid output resolution

Whenever invalid input is detected, meaningful exceptions are raised to assist debugging and improve robustness.

---

# Output

The **ComparisonVisualizer** produces publication-quality Matplotlib figures suitable for:

- Thesis documentation
- Journal publications
- Conference papers
- Qualitative experiment analysis
- Technical reports

All visualizations are returned as **`matplotlib.figure.Figure`** objects, allowing additional customization if required.

---

# Current Workflow

At the current stage of development, the visualization workflow is independent of any segmentation model.

```text
CHAOS Dataset
        │
        ▼
DICOM Reader
        │
        ▼
GroundTruthReader
        │
        ▼
ComparisonVisualizer
        │
        ▼
Qualitative Visualization
```

Deterministic synthetic prediction masks are currently used during testing to validate the visualization pipeline independently of neural network inference.

This ensures that the visualization infrastructure is fully validated before model implementation begins.

---

# Future Integration

Following implementation of the segmentation architectures, the workflow remains unchanged.

The only difference is that synthetic prediction masks will be replaced with actual model outputs.

```text
CHAOS Dataset
        │
        ▼
Preprocessing
        │
        ▼
Segmentation Model
        │
        ▼
Prediction Mask
        │
        ▼
ComparisonVisualizer
        │
        ▼
Qualitative Comparison
```

This architecture-independent design enables seamless integration with future segmentation models.

---

# Supported Segmentation Models

The **ComparisonVisualizer** is designed to support every segmentation architecture implemented within this research framework.

Planned models include:

- U-Net
- Attention U-Net
- 3D U-Net
- TransUNet
- UNETR
- Swin UNETR
- Proposed Hybrid Multimodal Architecture

Since the module operates on generic prediction masks represented as NumPy arrays, no implementation changes are required when integrating additional segmentation architectures.

---

# Future Extensions

Several enhancements are planned for future versions of the module.

## Multi-Class Segmentation

Support qualitative comparison of multiple anatomical structures using distinct color maps.

---

## Contour Visualization

Display predicted and ground truth contours simultaneously to improve boundary analysis.

---

## Error Visualization

Differentiate:

- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)

using dedicated color coding rather than a binary difference map.

---

## Interactive Visualization

Support interactive exploration using visualization libraries such as:

- Plotly
- Napari

---

## 3D Volume Visualization

Extend comparison capabilities from individual image slices to complete volumetric CT and MRI datasets.

---

## Confidence Visualization

Display prediction confidence and uncertainty maps generated by segmentation models.

---

## Automatic Report Generation

Generate standardized qualitative comparison reports suitable for:

- Thesis chapters
- Research publications
- Benchmark reports

---

# Testing Strategy

The **ComparisonVisualizer** is validated using both unit testing and integration testing.

## Unit Testing

Unit tests verify:

- Figure creation
- Comparison layouts
- Overlay generation
- Difference visualization
- Figure export
- Input validation
- Exception handling

Synthetic deterministic NumPy arrays are used to ensure reproducibility and fast execution.

---

## Integration Testing

Integration tests use real CHAOS CT images together with corresponding ground truth segmentation masks.

These tests verify:

- Complete visualization pipeline
- Compatibility with real medical images
- Correct rendering of qualitative comparisons
- Independence from segmentation model inference

Once segmentation architectures are implemented, these tests will be extended to use real prediction masks generated by each model.

---

# Advantages

The **ComparisonVisualizer** provides several advantages.

- Architecture-independent implementation
- Reusable across multiple segmentation models
- Publication-quality visualization
- Modular and maintainable design
- Comprehensive input validation
- Robust exception handling
- Consistent visualization workflow
- Seamless integration with the medical image segmentation framework
- Easy extensibility for future qualitative analysis techniques
- Foundation for systematic error analysis and research gap identification

---

# Conclusion

The **ComparisonVisualizer** is a core component of the visualization subsystem within the medical image segmentation framework.

By providing standardized qualitative comparisons across multiple segmentation architectures, it enables researchers to move beyond numerical evaluation metrics and investigate **how**, **where**, and **why** different models succeed or fail.

Its reusable architecture, comprehensive validation, publication-quality outputs, and independence from specific segmentation models make it an essential tool for benchmarking, qualitative evaluation, error analysis, and research gap discovery.

Together with the **BaseVisualizer**, **MaskVisualizer**, and **OverlayVisualizer**, the **ComparisonVisualizer** establishes a complete visualization framework that supports reproducible experimentation and evidence-based medical image segmentation research.