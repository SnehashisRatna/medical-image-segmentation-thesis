"""Partition medical image samples into train/validation/test subsets."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from src.core.enums import Modality
from src.datasets.sample import Sample


@dataclass(frozen=True, slots=True)
class SplitConfiguration:
    """Configure the deterministic splitting of patient identities.

    Parameters
    ----------
    train_ratio : float
        Ratio of patients allocated to the training split.
    validation_ratio : float
        Ratio of patients allocated to the validation split.
    test_ratio : float
        Ratio of patients allocated to the test split.
    seed : int
        Random seed for deterministic assignment.
    """

    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify_by_modality: bool = False

    def __post_init__(self) -> None:
        """Validate split ratios."""
        if self.train_ratio < 0 or self.validation_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative.")
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Mutually exclusive collections of Samples.

    Parameters
    ----------
    train : list[Sample]
        Samples in the training split.
    validation : list[Sample]
        Samples in the validation split.
    test : list[Sample]
        Samples in the test split.
    """

    train: list[Sample]
    validation: list[Sample]
    test: list[Sample]


class DatasetSplitter:
    """Partition Sample references into train/validation/test subsets.

    Splitting is performed at the PATIENT identity level. All samples for a
    (modality, patient_id) are guaranteed to fall into the same split, avoiding
    leakage. The assignment process is completely deterministic based on the
    configuration's random seed. Modality stratification is optional and controlled
    through SplitConfiguration.stratify_by_modality.

    Parameters
    ----------
    config : SplitConfiguration
        Configuration specifying ratios and deterministic random seed.
    """

    def __init__(self, config: SplitConfiguration) -> None:
        self.config = config

    def split(self, samples: Sequence[Sample]) -> DatasetSplit:
        """Partition samples according to configured ratios.

        Parameters
        ----------
        samples : Sequence[Sample]
            The source sequence of samples to split.

        Returns
        -------
        DatasetSplit
            A validated split result object containing train, validation, and test
            subsets.

        Raises
        ------
        ValueError
            If the supplied sample collection cannot be safely split or ratios are invalid.
        """
        if not samples:
            raise ValueError("Cannot split an empty sequence of samples.")

        patient_samples: defaultdict[tuple[Modality, str], list[Sample]] = defaultdict(
            list
        )
        for sample in samples:
            patient_samples[(sample.modality, sample.patient_id)].append(sample)

        train_samples: list[Sample] = []
        val_samples: list[Sample] = []
        test_samples: list[Sample] = []

        rng = random.Random(self.config.seed)

        if self.config.stratify_by_modality:
            modality_patients: defaultdict[Modality, list[tuple[Modality, str]]] = (
                defaultdict(list)
            )
            for key in patient_samples.keys():
                modality_patients[key[0]].append(key)

            for modality in sorted(modality_patients.keys(), key=lambda m: m.name):
                patient_keys = sorted(
                    modality_patients[modality], key=lambda k: (k[0].name, k[1])
                )
                t_keys, v_keys, test_k = self._split_keys(patient_keys, rng)
                for key in t_keys:
                    train_samples.extend(patient_samples[key])
                for key in v_keys:
                    val_samples.extend(patient_samples[key])
                for key in test_k:
                    test_samples.extend(patient_samples[key])
        else:
            patient_keys = sorted(
                list(patient_samples.keys()), key=lambda k: (k[0].name, k[1])
            )
            t_keys, v_keys, test_k = self._split_keys(patient_keys, rng)
            for key in t_keys:
                train_samples.extend(patient_samples[key])
            for key in v_keys:
                val_samples.extend(patient_samples[key])
            for key in test_k:
                test_samples.extend(patient_samples[key])

        result = DatasetSplit(
            train=train_samples, validation=val_samples, test=test_samples
        )

        self._validate_result(result, samples)

        return result

    def _split_keys(
        self, patient_keys: list[tuple[Modality, str]], rng: random.Random
    ) -> tuple[
        list[tuple[Modality, str]],
        list[tuple[Modality, str]],
        list[tuple[Modality, str]],
    ]:
        rng.shuffle(patient_keys)

        n_patients = len(patient_keys)
        n_train = round(n_patients * self.config.train_ratio)
        n_val = round(n_patients * self.config.validation_ratio)
        n_test = round(n_patients * self.config.test_ratio)

        diff = n_patients - (n_train + n_val + n_test)
        if diff != 0:
            if (
                self.config.train_ratio >= self.config.validation_ratio
                and self.config.train_ratio >= self.config.test_ratio
            ):
                n_train += diff
            elif self.config.validation_ratio >= self.config.test_ratio:
                n_val += diff
            else:
                n_test += diff

        if self.config.train_ratio > 0 and n_train == 0:
            raise ValueError(
                f"Requested train_ratio={self.config.train_ratio} cannot be satisfied with {n_patients} patient(s)."
            )
        if self.config.validation_ratio > 0 and n_val == 0:
            raise ValueError(
                f"Requested validation_ratio={self.config.validation_ratio} cannot be satisfied with {n_patients} patient(s)."
            )
        if self.config.test_ratio > 0 and n_test == 0:
            raise ValueError(
                f"Requested test_ratio={self.config.test_ratio} cannot be satisfied with {n_patients} patient(s)."
            )

        train_ids = patient_keys[:n_train]
        val_ids = patient_keys[n_train : n_train + n_val]
        test_ids = patient_keys[n_train + n_val :]

        return train_ids, val_ids, test_ids

    def _validate_result(
        self, result: DatasetSplit, original: Sequence[Sample]
    ) -> None:
        """Validate patient exclusivity, sample conservation, and no duplicates."""
        # Patient exclusivity
        train_patients = self._get_patients(result.train)
        val_patients = self._get_patients(result.validation)
        test_patients = self._get_patients(result.test)

        if train_patients & val_patients:
            raise ValueError(
                "Patient leakage detected between train and validation splits."
            )
        if train_patients & test_patients:
            raise ValueError("Patient leakage detected between train and test splits.")
        if val_patients & test_patients:
            raise ValueError(
                "Patient leakage detected between validation and test splits."
            )

        # Duplicate checking within splits
        self._check_duplicates(result.train, "train")
        self._check_duplicates(result.validation, "validation")
        self._check_duplicates(result.test, "test")

        # Sample conservation
        from collections import Counter

        if Counter(result.train + result.validation + result.test) != Counter(original):
            raise ValueError(
                "Sample conservation check failed: output does not contain every original Sample exactly once."
            )

    @staticmethod
    def _get_patients(samples: list[Sample]) -> set[tuple[Modality, str]]:
        return {(s.modality, s.patient_id) for s in samples}

    @staticmethod
    def _check_duplicates(samples: list[Sample], split_name: str) -> None:
        if len(samples) != len(set(samples)):
            raise ValueError(f"Duplicate samples found in {split_name} split.")
