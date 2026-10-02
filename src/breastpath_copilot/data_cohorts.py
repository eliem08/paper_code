"""Multi-cohort dataset parsers for the early-detection Q1 extension.

Adds BACH (ICIAR 2018) and BRACS (RoI release) to the BreaKHis support already
in :mod:`breastpath_copilot.data`. All cohorts are normalised to a shared
early-detection label vocabulary so that downstream MoE / retrieval / agentic
tools can train and evaluate cross-cohort under leave-one-cohort-out.

Unified label vocabulary (mirrors :data:`configs/default_config.yaml`)::

    normal | benign | atypia | dcis | invasive

The per-cohort native subtype is preserved in the ``subtype`` column for
lesion-stratified reporting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

from breastpath_copilot.data import (
    BENIGN_SUBTYPE_MAP,
    MALIGNANT_SUBTYPE_MAP,
    is_image_file,
    iter_breakhis_images,
)

UNIFIED_LABELS = ("normal", "benign", "atypia", "dcis", "invasive")

# --------------------------------------------------------------------------- #
# BACH (ICIAR 2018)
# --------------------------------------------------------------------------- #

BACH_DIR_TO_LABEL = {
    "normal": "normal",
    "benign": "benign",
    "insitu": "dcis",
    "in_situ": "dcis",
    "invasive": "invasive",
}

BACH_DIR_TO_SUBTYPE = {
    "normal": "normal",
    "benign": "benign",
    "insitu": "in_situ_carcinoma",
    "in_situ": "in_situ_carcinoma",
    "invasive": "invasive_carcinoma",
}


@dataclass(frozen=True)
class BachPathInfo:
    image_path: Path
    image_id: str
    label: str
    subtype: str


def _normalise_bach_dir(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def parse_bach_image_path(image_path: str | Path) -> BachPathInfo:
    path = Path(image_path)
    if not is_image_file(path):
        raise ValueError(f"Not a supported image file: {path}")
    parent = _normalise_bach_dir(path.parent.name)
    if parent not in BACH_DIR_TO_LABEL:
        raise ValueError(
            f"Unexpected BACH class folder {parent!r} (expected one of {list(BACH_DIR_TO_LABEL)})"
        )
    return BachPathInfo(
        image_path=path,
        image_id=path.stem,
        label=BACH_DIR_TO_LABEL[parent],
        subtype=BACH_DIR_TO_SUBTYPE[parent],
    )


def iter_bach_images(root_dir: str | Path) -> Iterable[BachPathInfo]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"BACH root not found: {root}")
    for path in root.rglob("*"):
        if not is_image_file(path):
            continue
        try:
            yield parse_bach_image_path(path)
        except ValueError:
            continue


def build_bach_manifest(root_dir: str | Path) -> pd.DataFrame:
    rows = []
    for info in iter_bach_images(root_dir):
        rows.append(
            {
                "image_path": str(info.image_path),
                "image_id": f"bach_{info.image_id}",
                "patient_id": f"bach_{info.image_id}",  # one ROI per pseudo-patient
                "label": info.label,
                "subtype": info.subtype,
                "magnification": "unknown",
                "cohort": "bach",
            }
        )
    if not rows:
        raise ValueError(f"No BACH images found under: {root_dir}")
    return (
        pd.DataFrame(rows)
        .sort_values(["label", "subtype", "image_id"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# BRACS (RoI release)
# --------------------------------------------------------------------------- #

# BRACS native class folders. The official BRACS_RoI release names folders with
# a numeric prefix (0_N, 1_PB, ..., 6_IC); we also accept the bare abbreviations
# and the fully-spelled-out variants.
_BRACS_RAW_LABELS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("0_n", "n", "normal"),                       "normal",   "normal"),
    (("1_pb", "pb", "pathological_benign", "benign"),     "benign",   "pathological_benign"),
    (("2_udh", "udh", "usual_ductal_hyperplasia"),        "benign",   "usual_ductal_hyperplasia"),
    (("3_fea", "fea", "flat_epithelial_atypia"),          "atypia",   "flat_epithelial_atypia"),
    (("4_adh", "adh", "atypical_ductal_hyperplasia"),     "atypia",   "atypical_ductal_hyperplasia"),
    (("5_dcis", "dcis", "ductal_carcinoma_in_situ"),      "dcis",     "ductal_carcinoma_in_situ"),
    (("6_ic", "ic", "invasive_carcinoma", "invasive"),    "invasive", "invasive_carcinoma"),
)

BRACS_DIR_TO_LABEL: dict[str, str] = {alias: label for aliases, label, _ in _BRACS_RAW_LABELS for alias in aliases}
BRACS_DIR_TO_SUBTYPE: dict[str, str] = {alias: subtype for aliases, _, subtype in _BRACS_RAW_LABELS for alias in aliases}


@dataclass(frozen=True)
class BracsPathInfo:
    image_path: Path
    image_id: str
    patient_id: str
    label: str
    subtype: str
    split: str | None = None


_BRACS_PATIENT_RE = re.compile(r"^(?:BRACS[_-]?)?(\d+)", re.IGNORECASE)
_BRACS_SPLIT_NAMES: dict[str, str] = {
    "train": "train",
    "test": "test",
    "val": "val",
    "validation": "val",
}


def _extract_bracs_patient_id(stem: str) -> str:
    m = _BRACS_PATIENT_RE.search(stem)
    return f"bracs_{m.group(1)}" if m else f"bracs_{stem}"


def _normalise_bracs_dir(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _extract_bracs_split(path: Path) -> str | None:
    """Look for train/test/val tokens anywhere in the path ancestry."""
    for part in path.parts:
        normalised = part.strip().lower()
        if normalised in _BRACS_SPLIT_NAMES:
            return _BRACS_SPLIT_NAMES[normalised]
    return None


def parse_bracs_image_path(image_path: str | Path) -> BracsPathInfo:
    path = Path(image_path)
    if not is_image_file(path):
        raise ValueError(f"Not a supported image file: {path}")
    parent = _normalise_bracs_dir(path.parent.name)
    if parent not in BRACS_DIR_TO_LABEL:
        raise ValueError(
            f"Unexpected BRACS class folder {parent!r}; expected one of {sorted(set(BRACS_DIR_TO_LABEL))}"
        )
    return BracsPathInfo(
        image_path=path,
        image_id=f"bracs_{path.stem}",
        patient_id=_extract_bracs_patient_id(path.stem),
        label=BRACS_DIR_TO_LABEL[parent],
        subtype=BRACS_DIR_TO_SUBTYPE[parent],
        split=_extract_bracs_split(path),
    )


BRACS_VERSION_TYPE = Literal["latest", "v1", "all"]


def _resolve_bracs_walk_root(root: Path, version: str) -> Path:
    """Pick which subtree of the BRACS_RoI root to walk.

    The official release lays out::

        BRACS_RoI/
            latest_version/{train,test,val}/{0_N..6_IC}/*.png
            previous_versions/Version1_MedIA/Images/{train,test,val}/{...}/*.png

    Walking the whole tree double-counts every case. We default to the
    most recent release; pass ``version='all'`` to opt back in to both.
    """
    if version == "all":
        return root
    latest = root / "latest_version"
    v1 = root / "previous_versions" / "Version1_MedIA" / "Images"
    if version == "latest":
        if latest.exists():
            return latest
        # Already pointed at a single-version subtree -> walk as-is.
        return root
    if version == "v1":
        if v1.exists():
            return v1
        return root
    raise ValueError(
        f"Unknown BRACS version {version!r}. Expected one of: latest, v1, all"
    )


def iter_bracs_images(
    root_dir: str | Path,
    version: BRACS_VERSION_TYPE = "latest",
) -> Iterable[BracsPathInfo]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"BRACS root not found: {root}")
    walk_root = _resolve_bracs_walk_root(root, version)
    for path in walk_root.rglob("*"):
        if not is_image_file(path):
            continue
        try:
            yield parse_bracs_image_path(path)
        except ValueError:
            continue


def build_bracs_manifest(
    root_dir: str | Path,
    version: BRACS_VERSION_TYPE = "latest",
) -> pd.DataFrame:
    rows = []
    for info in iter_bracs_images(root_dir, version=version):
        rows.append(
            {
                "image_path": str(info.image_path),
                "image_id": info.image_id,
                "patient_id": info.patient_id,
                "label": info.label,
                "subtype": info.subtype,
                "magnification": "unknown",
                "cohort": "bracs",
                "split": info.split,
            }
        )
    if not rows:
        raise ValueError(f"No BRACS images found under: {root_dir}")
    return (
        pd.DataFrame(rows)
        .sort_values(["split", "label", "subtype", "patient_id", "image_id"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# BreaKHis -> unified
# --------------------------------------------------------------------------- #

BREAKHIS_SUBTYPE_TO_LABEL = {
    # Benign subtypes
    **{k: "benign" for k in BENIGN_SUBTYPE_MAP},
    # Malignant subtypes are all invasive carcinomas in BreaKHis (no DCIS labels).
    **{k: "invasive" for k in MALIGNANT_SUBTYPE_MAP},
}


def build_breakhis_unified_manifest(root_dir: str | Path) -> pd.DataFrame:
    rows = []
    for info in iter_breakhis_images(root_dir):
        rows.append(
            {
                "image_path": str(info.image_path),
                "image_id": f"breakhis_{info.image_id}",
                "patient_id": f"breakhis_{info.patient_id}",
                "label": BREAKHIS_SUBTYPE_TO_LABEL.get(info.subtype, info.label),
                "subtype": info.subtype,
                "magnification": info.magnification,
                "cohort": "breakhis",
            }
        )
    if not rows:
        raise ValueError(f"No BreaKHis images found under: {root_dir}")
    return (
        pd.DataFrame(rows)
        .sort_values(["label", "subtype", "patient_id", "image_id"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Cross-cohort utilities
# --------------------------------------------------------------------------- #

COHORT_BUILDERS = {
    "breakhis": build_breakhis_unified_manifest,
    "bach": build_bach_manifest,
    "bracs": build_bracs_manifest,
}


def build_cross_cohort_manifest(cohort_roots: dict[str, str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, root in cohort_roots.items():
        key = name.lower()
        if key not in COHORT_BUILDERS:
            raise KeyError(f"Unknown cohort {name!r}. Available: {sorted(COHORT_BUILDERS)}")
        frames.append(COHORT_BUILDERS[key](root))
    if not frames:
        return pd.DataFrame(
            columns=["image_path", "image_id", "patient_id", "label", "subtype", "magnification", "cohort"]
        )
    return pd.concat(frames, ignore_index=True)


def leave_one_cohort_out_splits(
    manifest: pd.DataFrame,
) -> Iterable[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(held_out_cohort, train_df, test_df)`` triples.

    Train pool is the union of all other cohorts; test is the held-out cohort.
    Caller is responsible for further patient-aware splitting inside the train pool.
    """
    if "cohort" not in manifest.columns:
        raise ValueError("Manifest must contain a 'cohort' column")
    cohorts = sorted(manifest["cohort"].astype(str).unique())
    for held in cohorts:
        mask = manifest["cohort"].astype(str) == held
        yield held, manifest.loc[~mask].reset_index(drop=True), manifest.loc[mask].reset_index(drop=True)
