# Sample

## Immutability / Data Integrity

`Sample` should be treated as a description of a dataset item rather than a mutable processing object.

Once created, its identity-related properties should not be casually modified during preprocessing or model execution.

For example:

```text
DatasetIndexer
      │
      ▼
   Sample
      │
      ├── image reference
      ├── mask reference
      ├── patient
      └── modality
```

should remain stable while:

```text
Preprocessing
      ↓
Normalization
      ↓
Augmentation
```

operates on the loaded data rather than modifying the sample's identity.

---

# Separation from DICOM Reader

`Sample` should not depend on the implementation details of the DICOM reader.

The responsibilities are deliberately separated:

### DICOM Reader

```text
DICOM file
    ↓
DICOMReader
    ↓
image / metadata
```

### Sample

```text
image reference
mask reference
patient
modality
metadata
```

### DatasetIndexer

```text
Dataset directory
       ↓
discover files
       ↓
match image ↔ mask
       ↓
create Sample objects
```

This prevents `Sample` from becoming tightly coupled to DICOM-specific logic.

---

# Relationship with DatasetIndexer

`DatasetIndexer` is expected to be the primary producer of `Sample` objects.

Conceptually:

```text
samples = indexer.index()
```

returns:

```text
list[Sample]
```

For example:

```text
DatasetIndexer
│
├── CT
│   ├── Patient 1 → Sample
│   ├── Patient 2 → Sample
│   └── ...
│
└── MRI
    ├── Patient 1 → Sample
    ├── Patient 2 → Sample
    └── ...
```

The indexer handles the complexity of discovering and pairing the raw dataset.

`Sample` simply represents the resulting item.

---

# Relationship with Future Components

The intended dependency direction is:

```text
Dataset
  │
  ▼
DatasetIndexer
  │
  ▼
Sample
  │
  ├──────────────► DataLoader
  │
  ├──────────────► Preprocessor
  │
  ├──────────────► Dataset
  │
  └──────────────► Training Pipeline
```

Higher-level components may consume `Sample`, but `Sample` should not depend on those higher-level components.

This keeps the architecture modular and testable.

---

# Example Conceptual Usage

```text
sample = Sample(
    ...
)
```

The sample can then be passed through the dataset pipeline:

```text
Sample
  ↓
Load image
  ↓
Load mask
  ↓
Preprocess
  ↓
Tensor conversion
  ↓
Model input
```

Importantly, creating a `Sample` does **not** imply that the medical image itself has been loaded into memory.

---

# Testing Strategy

The `Sample` class is tested independently from the complete dataset.

The tests verify:

- Correct construction
- Required fields
- Optional fields
- Modality handling
- Image reference handling
- Ground-truth handling
- Metadata preservation
- Equality/representation behavior where applicable
- Invalid input handling
- Expected object semantics

The current project regression suite confirms that the implementation integrates successfully with the existing architecture:

```text
192 passed in 8.35s
```

This means the `Sample` implementation has been introduced without breaking the existing visualization, core, dataset, and integration tests.

---

# Design Constraints

The following constraints should remain part of the `Sample` contract:

| Constraint | Requirement |
|---|---|
| Single Responsibility | Represent one dataset sample |
| DICOM independence | No DICOM parsing logic |
| Lightweight | Do not eagerly load image volumes |
| Modality aware | CT/MRI must be distinguishable |
| Ground truth | May or may not be available |
| Metadata | Preserve sample-level information |
| Extensibility | Support future modalities/datasets |
| Reusability | Usable by training and evaluation pipelines |
| Testability | Independently unit-testable |

---

# Why `Sample` Is Important

Although `Sample` is a relatively small component, it establishes an important architectural boundary.

Without a common sample representation, downstream components would need to understand the directory structure of every dataset:

```text
Training
   ↓
knows CHAOS structure
   ↓
Preprocessing
   ↓
knows CHAOS structure
   ↓
Evaluation
   ↓
knows CHAOS structure
```

With `Sample`:

```text
CHAOS
  ↓
DatasetIndexer
  ↓
Sample
  ↓
Training
Evaluation
Visualization
Inference
```

Only the dataset-specific indexing layer needs to understand the raw dataset organization.

This makes the framework much easier to extend to another medical segmentation dataset later.

---

# Future Extension

The `Sample` abstraction is intentionally designed to support the later stages of the thesis.

Potential future consumers include:

```text
Sample
  │
  ├── 2D U-Net
  ├── 3D U-Net
  ├── Attention U-Net
  ├── TransUNet
  ├── UNETR
  └── Swin UNETR
```

This is especially important because architectures such as 3D U-Net and UNETR operate on volumetric medical data rather than treating every image as an isolated natural-image sample. 3D U-Net explicitly extends U-Net to volumetric inputs using 3D operations, while UNETR reformulates volumetric segmentation using transformer-based sequence representations.

Therefore, `Sample` should remain **model-agnostic**.

---

# Summary

`Sample` provides the canonical representation of one medical-image segmentation sample in the thesis framework.

Its core principle is:

> **Discover the data once, represent it consistently, and keep all downstream components independent of the raw dataset structure.**

The resulting architecture is:

```text
Raw Medical Dataset
        │
        ▼
 DatasetIndexer
        │
        ▼
      Sample
        │
        ├── Data Loading
        ├── Preprocessing
        ├── Visualization
        ├── Training
        ├── Evaluation
        └── Inference
```

`Sample` is therefore a foundational component of the dataset infrastructure, while deliberately remaining independent of DICOM parsing, preprocessing, visualization, and model-specific logic.