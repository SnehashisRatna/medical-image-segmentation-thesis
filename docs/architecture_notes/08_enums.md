# Modality Enumeration

## Overview

The `Modality` enumeration defines the supported medical imaging modalities used throughout the Medical Image Segmentation Research Framework.

Instead of using raw strings, the framework uses an enumeration to provide consistency, type safety, and improved readability.

---

# Purpose

Medical datasets contain different imaging modalities.

Examples include:

- Computed Tomography (CT)
- Magnetic Resonance Imaging (MRI)
- Positron Emission Tomography (PET)

Using an enumeration eliminates spelling mistakes and ensures every module interprets imaging modalities consistently.

---

# Supported Modalities

| Enum | Description |
|------|-------------|
| CT | Computed Tomography |
| MRI | Magnetic Resonance Imaging |
| PET | Positron Emission Tomography |

---

# Implementation

```python
class Modality(Enum):
    CT = "CT"
    MRI = "MRI"
    PET = "PET"
```

---

# Why an Enumeration?

Using an enumeration provides several advantages over plain strings.

### Type Safety

```python
Modality.CT
```

instead of

```python
"CT"
```

---

### Consistency

Every module uses the same predefined values.

No accidental values such as

```
Ct
ct
MRI Scan
ComputedTomography
```

---

### Readability

The code becomes more expressive.

```python
if record.modality == Modality.MRI:
```

instead of

```python
if record.modality == "MRI":
```

---

### IDE Support

Most modern IDEs provide:

- Autocomplete
- Refactoring support
- Static analysis
- Type checking

---

# Example Usage

```python
record = ExperimentRecord(
    modality=Modality.CT,
    ...
)
```

---

# Design Principles

The enumeration follows:

- Explicit representation
- Strong typing
- Framework-wide consistency
- Future extensibility

---

# Testing

The enumeration is verified through unit tests covering:

- Member values
- Member names
- Number of supported modalities

---

# Future Extensions

Additional modalities may include:

- Ultrasound (US)
- X-Ray
- OCT
- Histopathology
- SPECT
- Functional MRI (fMRI)

without changing the existing framework architecture.

---

# Module Location

```
src/core/enums.py
```

---

# Dependencies

- Python Enum