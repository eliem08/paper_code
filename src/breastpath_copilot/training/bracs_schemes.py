"""Canonical BRACS label schemes for apples-to-apples literature comparison.

BRACS RoIs carry a fine-grained ``subtype`` (set by the parser in
:mod:`breastpath_copilot.data_cohorts`). From that single column we can
materialise three label schemes without re-extracting embeddings:

* **7-class** (the original Brancati et al. benchmark): N, PB, UDH, FEA, ADH,
  DCIS, IC.
* **3-class** (the widely-cited risk grouping): benign={N,PB,UDH},
  atypical={FEA,ADH}, malignant={DCIS,IC}.
* **5-class** (our early-detection scheme, defined in datasets.py): normal,
  benign, atypia, dcis, invasive.

Each scheme also defines its "needs attention" positive set so the binary
early-detection collapse (AUROC, sensitivity@specificity) is comparable
across schemes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from breastpath_copilot.training.datasets import UNIFIED_LABEL_TO_INDEX


# Map BRACS fine-grained subtype -> native 7-class code.
_SUBTYPE_TO_7CLASS = {
    "normal": "N",
    "pathological_benign": "PB",
    "usual_ductal_hyperplasia": "UDH",
    "flat_epithelial_atypia": "FEA",
    "atypical_ductal_hyperplasia": "ADH",
    "ductal_carcinoma_in_situ": "DCIS",
    "invasive_carcinoma": "IC",
}

# 7-class -> 3-class risk grouping.
_SEVEN_TO_THREE = {
    "N": "benign", "PB": "benign", "UDH": "benign",
    "FEA": "atypical", "ADH": "atypical",
    "DCIS": "malignant", "IC": "malignant",
}


@dataclass(frozen=True)
class LabelScheme:
    """A label scheme: name, label->index map, and the attention-positive set."""

    name: str
    label_to_index: dict[str, int]
    attention_positive_labels: tuple[str, ...]
    subtype_to_label: dict[str, str]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    @property
    def index_to_label(self) -> dict[int, str]:
        return {v: k for k, v in self.label_to_index.items()}

    @property
    def attention_positive_indices(self) -> list[int]:
        return [self.label_to_index[l] for l in self.attention_positive_labels if l in self.label_to_index]


SCHEME_7CLASS = LabelScheme(
    name="bracs_7class",
    label_to_index={"N": 0, "PB": 1, "UDH": 2, "FEA": 3, "ADH": 4, "DCIS": 5, "IC": 6},
    attention_positive_labels=("FEA", "ADH", "DCIS", "IC"),
    subtype_to_label=dict(_SUBTYPE_TO_7CLASS),
)

SCHEME_3CLASS = LabelScheme(
    name="bracs_3class",
    label_to_index={"benign": 0, "atypical": 1, "malignant": 2},
    attention_positive_labels=("atypical", "malignant"),
    subtype_to_label={st: _SEVEN_TO_THREE[code] for st, code in _SUBTYPE_TO_7CLASS.items()},
)

SCHEME_5CLASS = LabelScheme(
    name="bracs_5class",
    label_to_index=dict(UNIFIED_LABEL_TO_INDEX),
    attention_positive_labels=("atypia", "dcis", "invasive"),
    # 5-class label already lives in the manifest 'label' column; identity map
    # from subtype is provided for completeness.
    subtype_to_label={
        "normal": "normal",
        "pathological_benign": "benign",
        "usual_ductal_hyperplasia": "benign",
        "flat_epithelial_atypia": "atypia",
        "atypical_ductal_hyperplasia": "atypia",
        "ductal_carcinoma_in_situ": "dcis",
        "invasive_carcinoma": "invasive",
    },
)

SCHEMES: dict[str, LabelScheme] = {
    "3class": SCHEME_3CLASS,
    "5class": SCHEME_5CLASS,
    "7class": SCHEME_7CLASS,
}


def get_scheme(name: str) -> LabelScheme:
    key = name.lower().replace("-", "").replace("_", "")
    key = {"3class": "3class", "5class": "5class", "7class": "7class"}.get(key, key)
    if key not in SCHEMES:
        raise KeyError(f"Unknown scheme {name!r}. Available: {sorted(SCHEMES)}")
    return SCHEMES[key]


def remap_bracs_manifest(manifest: pd.DataFrame, scheme: LabelScheme) -> pd.DataFrame:
    """Return a copy of a BRACS manifest with ``label`` set to the scheme's labels.

    Requires a ``subtype`` column. Rows whose subtype is unmapped are dropped.
    """
    if "subtype" not in manifest.columns:
        raise ValueError("BRACS manifest must carry a 'subtype' column to remap labels")
    out = manifest.copy()
    out["label"] = out["subtype"].astype(str).map(scheme.subtype_to_label)
    before = len(out)
    out = out[out["label"].notna()].reset_index(drop=True)
    if len(out) < before:
        # Some subtype wasn't in the scheme map -- surface it rather than silently drop many.
        dropped = before - len(out)
        if dropped > 0.01 * before:
            raise ValueError(
                f"remap_bracs_manifest dropped {dropped}/{before} rows -- unmapped subtypes: "
                f"{sorted(set(manifest['subtype']) - set(scheme.subtype_to_label))}"
            )
    return out
