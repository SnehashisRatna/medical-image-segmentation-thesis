# Metadata Scanner

> **Module:** Metadata Scanner  
> **Project:** Multimodal Medical Image Segmentation Framework  
> **Dataset:** CHAOS (CT & MRI) Dataset

---

# 1. Overview

## Purpose

The **Metadata Scanner** is the first module in our medical image segmentation framework. It is responsible for automatically inspecting every imaging study within the **CHAOS dataset**, extracting important acquisition parameters from CT and MRI DICOM files, and generating a structured metadata report.

Rather than manually examining hundreds of DICOM files, this module automates the process of collecting scanner information, imaging protocols, spatial resolution, slice thickness, and other acquisition characteristics required for downstream preprocessing and model development.

The generated metadata serves as the foundation for understanding dataset heterogeneity and enables informed preprocessing decisions before any deep learning model is trained.

### Inputs

- CHAOS Dataset
- CT DICOM Series
- MRI DICOM Series
  - T1DUAL InPhase
  - T1DUAL OutPhase
  - T2SPIR

### Outputs

- `dataset_summary.csv`
- Patient-wise metadata report
- Acquisition parameter summary

---

# 2. Motivation

Medical image datasets are inherently heterogeneous. Images acquired from different scanners, imaging protocols, slice thicknesses, reconstruction kernels, resolutions, and acquisition settings often exhibit significant variability.

Such heterogeneity directly impacts the performance and generalization capability of deep learning models.

Manually inspecting every imaging study is both time-consuming and error-prone. Therefore, we developed an automated **Metadata Scanner** capable of systematically characterizing the entire CHAOS dataset before preprocessing.

The scanner provides a reproducible and scalable approach for dataset exploration, allowing researchers to identify variations in acquisition parameters and design preprocessing strategies accordingly.

By automating this process, we ensure consistency, reduce manual effort, and establish a reliable quality assurance step within the overall segmentation framework.

---

# 3. Research Context

Modern medical image segmentation frameworks emphasize comprehensive dataset analysis prior to preprocessing.

Frameworks such as:

- MONAI
- nnU-Net
- TotalSegmentator

perform automatic inspection of image properties before initiating preprocessing pipelines.

Inspired by these best practices, the Metadata Scanner provides automated acquisition analysis specifically designed for the CHAOS multimodal dataset.

The extracted metadata enables:

- Dataset characterization
- Identification of acquisition variability
- Validation of imaging protocols
- Support for preprocessing decisions
- Improved experimental reproducibility

This aligns our framework with contemporary research methodologies in medical image analysis.

---

# 4. Design Goals

The Metadata Scanner was designed with the following objectives.

## Primary Goals

- Fully automatic metadata extraction
- Reusable across multiple datasets
- Modality independent architecture
- Easily extensible
- Produce structured reports
- Minimize duplicated code
- Support future preprocessing modules
- Improve reproducibility

---

# 5. Architecture

```text
                    CHAOS Dataset
                          │
                          ▼
                 Metadata Scanner
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
     CT Scanner                     MRI Scanner
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 │                      │                      │
                 ▼                      ▼                      ▼
          T1DUAL_IN             T1DUAL_OUT               T2SPIR
                 │                      │                      │
                 └──────────────┬───────┴──────────────┬───────┘
                                ▼
                          DICOM Reader
                                │
                                ▼
                    Metadata Extraction
                                │
                                ▼
                     dataset_summary.csv
```

### Workflow

1. Load CHAOS dataset
2. Detect imaging modality
3. Traverse patient folders
4. Read DICOM metadata
5. Extract acquisition parameters
6. Store metadata records
7. Export structured CSV report

---

# 6. Design Decisions

## Why Separate CT and MRI Scanners?

CT and MRI possess fundamentally different directory structures and acquisition protocols.

Maintaining separate scanning methods provides:

- Better readability
- Reduced conditional complexity
- Easier maintenance
- Modality-specific extensions
- Improved scalability

This separation follows the principle of modular software design.

---

## Why Separate MRI Sequence Scanners?

MRI contains multiple acquisition protocols including:

- T1DUAL InPhase
- T1DUAL OutPhase
- T2SPIR

Each sequence possesses unique imaging characteristics.

Treating every sequence independently simplifies future expansion and sequence-specific preprocessing.

---

## Why Read Only Metadata?

The purpose of this module is dataset characterization.

Image preprocessing, normalization, augmentation, and segmentation are intentionally excluded to preserve modularity and maintain a single responsibility.

---

## Why Not Read Ground Truth Masks?

Ground truth masks represent segmentation annotations rather than acquisition metadata.

Including mask processing inside this module would violate the **Single Responsibility Principle (SRP)**.

Ground truth processing is therefore delegated to a dedicated **GroundTruth Reader** module.

---

## Why Export CSV?

CSV files provide:

- Human-readable reports
- Easy integration with Python
- Compatibility with Excel
- Statistical analysis using Pandas
- Documentation of acquisition parameters

CSV also serves as a reproducible record of dataset analysis.

---

# 7. Implementation Details

The Metadata Scanner follows a hierarchical execution flow.

```text
scan()
    │
    ▼
_scan_ct()
    │
    ▼
_scan_mr()
    │
    ├─────────────┐
    ▼             ▼
_scan_t1dual()   _scan_t2spir()
        │
        ▼
_scan_mr_sequence()
        │
        ▼
_extract_metadata()
        │
        ▼
save_csv()
```

## Method Responsibilities

### `scan()`

Main entry point responsible for initiating the complete metadata scanning pipeline.

---

### `_scan_ct()`

Traverses CT patient directories and extracts CT-specific acquisition metadata.

---

### `_scan_mr()`

Coordinates MRI scanning and delegates sequence-specific processing.

---

### `_scan_t1dual()`

Processes both InPhase and OutPhase MRI acquisitions.

---

### `_scan_t2spir()`

Processes T2SPIR MRI studies.

---

### `_scan_mr_sequence()`

Generalized method responsible for scanning any MRI sequence.

---

### `_extract_metadata()`

Reads DICOM headers and extracts:

- Patient ID
- Modality
- Sequence
- Slice count
- Pixel spacing
- Slice thickness
- Matrix dimensions
- Rescale slope
- Rescale intercept
- Bits stored
- Manufacturer information
- Additional acquisition parameters

---

### `save_csv()`

Generates the final structured metadata report.

---

# 8. Validation

The Metadata Scanner was validated using the complete CHAOS dataset.

## Integration Test Results

| Validation Item | Status |
|-----------------|--------|
| CT Patients | ✅ 20 |
| MRI Patients | ✅ 20 |
| MRI Sequences | ✅ 3 |
| Total Records | ✅ 80 |
| CSV Generation | ✅ Successful |
| Metadata Extraction | ✅ Successful |

### Validation Summary

The scanner successfully traversed the complete dataset without manual intervention and generated a structured metadata report for all imaging studies.

---

# 9. Research Findings

Analysis of the extracted metadata revealed several important observations.

## CT Observations

- Slice thickness varies approximately between **2–3.2 mm**
- Hounsfield Unit conversion depends on scanner-specific rescale parameters
- Pixel spacing remains relatively consistent
- Acquisition parameters vary across patients

## MRI Observations

- Slice thickness varies approximately between **5–8 mm**
- MRI does not require Hounsfield Unit conversion
- Different MRI sequences contain different slice counts
- Spatial resolution differs between imaging protocols

## Overall Findings

The CHAOS dataset exhibits multimodal acquisition variability that motivates standardized preprocessing before segmentation.

These findings directly influence subsequent preprocessing stages including:

- Image resampling
- Intensity normalization
- Spatial alignment
- Dataset harmonization

---

# 10. Future Improvements

Potential extensions include:

- Introduce a `DICOMMetadata` dataclass
- Add comprehensive logging
- Support additional DICOM attributes
- Generate statistical visualizations
- Export JSON metadata
- Automatic anomaly detection
- Dataset consistency validation
- Integration with preprocessing pipeline
- Interactive metadata dashboard

---

# 11. Contribution to Thesis

This module contributes directly to **Chapter 3: Dataset Analysis** of the thesis.

Its primary contributions include:

- Automated dataset characterization
- Quantitative analysis of acquisition parameters
- Identification of modality-specific variability
- Support for preprocessing design decisions
- Improved reproducibility of experiments

The generated metadata serves as empirical evidence supporting the preprocessing methodology presented later in the thesis.

---

# 12. Contribution to Future Research Paper

The Metadata Scanner establishes a reusable dataset analysis pipeline that extends beyond the CHAOS dataset.

Its modular architecture enables adaptation to additional CT and MRI datasets with minimal modification.

This component contributes toward:

- Reproducible medical image analysis
- Standardized dataset inspection
- Modular segmentation frameworks
- Future multimodal medical imaging research

The design can be incorporated into future publications describing scalable preprocessing pipelines and reproducible deep learning workflows.

---

# Key Contributions

- Automated metadata extraction
- Modular CT and MRI scanning framework
- Structured acquisition analysis
- Dataset characterization
- Reproducible quality assurance pipeline
- Foundation for preprocessing
- Support for thesis methodology
- Reusable research framework

---

# References

1. CHAOS Challenge Dataset
2. MONAI Framework
3. nnU-Net: Self-Adapting Framework for Medical Image Segmentation
4. TotalSegmentator
5. DICOM Standard Documentation

---

> **Documentation Philosophy**

Every major module in this project follows two complementary deliverables:

- **Production-quality implementation** located in `src/`
- **Research-quality documentation** located in `docs/`

The objective is to ensure that each software component is accompanied by comprehensive design documentation, architectural rationale, implementation details, validation results, and research insights. Over time, these documents collectively form the foundation of the thesis methodology, future research publications, and potential Ph.D. proposals, promoting reproducibility, maintainability, and scientific rigor.