from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

BENIGN_SUBTYPE_MAP = {
    'adenosis': 'adenosis',
    'fibroadenoma': 'fibroadenoma',
    'phyllodes_tumor': 'phyllodes_tumor',
    'tubular_adenoma': 'tubular_adenoma',
}

MALIGNANT_SUBTYPE_MAP = {
    'ductal_carcinoma': 'ductal_carcinoma',
    'lobular_carcinoma': 'lobular_carcinoma',
    'mucinous_carcinoma': 'mucinous_carcinoma',
    'papillary_carcinoma': 'papillary_carcinoma',
}

MAGNIFICATION_ORDER = ['40X', '100X', '200X', '400X']


@dataclass(frozen=True)
class BreakHisPathInfo:
    image_path: Path
    label: str
    superclass: str
    subtype: str
    magnification: str
    patient_id: str
    image_id: str


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS and (not path.exists() or path.is_file())


def parse_breakhis_image_path(image_path: str | Path) -> BreakHisPathInfo:
    path = Path(image_path)
    if not is_image_file(path):
        raise ValueError(f'Not a supported image file: {path}')

    parts = [p.lower() for p in path.parts]
    try:
        breast_idx = parts.index('breast')
    except ValueError as e:
        raise ValueError(f"Could not locate 'breast' folder in path: {path}") from e

    try:
        superclass = parts[breast_idx + 1]
        _sob = parts[breast_idx + 2]
        subtype = parts[breast_idx + 3]
        patient_id = path.parts[breast_idx + 4]
        magnification = path.parts[breast_idx + 5]
    except IndexError as e:
        raise ValueError(f'Unexpected BreaKHis folder layout for path: {path}') from e

    if superclass not in {'benign', 'malignant'}:
        raise ValueError(f'Unexpected class folder {superclass!r} in path: {path}')

    valid_subtypes = BENIGN_SUBTYPE_MAP if superclass == 'benign' else MALIGNANT_SUBTYPE_MAP
    if subtype not in valid_subtypes:
        raise ValueError(f'Unexpected subtype {subtype!r} in path: {path}')

    image_id = path.stem
    return BreakHisPathInfo(
        image_path=path,
        label=superclass,
        superclass=superclass,
        subtype=subtype,
        magnification=magnification,
        patient_id=patient_id,
        image_id=image_id,
    )


def iter_breakhis_images(root_dir: str | Path) -> Iterable[BreakHisPathInfo]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f'Dataset root not found: {root}')

    for path in root.rglob('*'):
        if not is_image_file(path):
            continue
        try:
            yield parse_breakhis_image_path(path)
        except ValueError:
            continue


def build_breakhis_manifest(root_dir: str | Path) -> pd.DataFrame:
    rows = []
    for info in iter_breakhis_images(root_dir):
        rows.append(
            {
                'image_path': str(info.image_path),
                'image_id': info.image_id,
                'patient_id': info.patient_id,
                'label': info.label,
                'superclass': info.superclass,
                'subtype': info.subtype,
                'magnification': info.magnification,
                'slide_id': derive_slide_id_from_path(info.image_path),
            }
        )

    if not rows:
        raise ValueError(f'No BreaKHis images found under: {root_dir}')

    df = pd.DataFrame(rows).sort_values(['label', 'subtype', 'patient_id', 'image_id']).reset_index(drop=True)
    return df


def derive_slide_id_from_path(image_path: str | Path) -> str:
    info = parse_breakhis_image_path(image_path)
    safe_patient = re.sub(r'[^A-Za-z0-9_-]+', '_', info.patient_id)
    safe_image = re.sub(r'[^A-Za-z0-9_-]+', '_', info.image_id)
    return f'{info.label}_{info.subtype}_{safe_patient}_{info.magnification}_{safe_image}'


def add_magnification_one_hot(df: pd.DataFrame, column: str = 'magnification') -> pd.DataFrame:
    out = df.copy()
    for mag in MAGNIFICATION_ORDER:
        out[f'mag_{mag.lower()}'] = (out[column].astype(str) == mag).astype(int)
    return out


def patient_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = 'patient_id',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[group_col].astype(str).values
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df


def summarize_manifest(df: pd.DataFrame) -> dict[str, object]:
    return {
        'num_images': int(len(df)),
        'num_patients': int(df['patient_id'].nunique()) if 'patient_id' in df else 0,
        'label_counts': df['label'].value_counts().sort_index().to_dict() if 'label' in df else {},
        'magnification_counts': df['magnification'].value_counts().sort_index().to_dict() if 'magnification' in df else {},
        'subtype_counts': df['subtype'].value_counts().sort_index().to_dict() if 'subtype' in df else {},
    }


def safe_auc_inputs(y_score: np.ndarray) -> np.ndarray:
    score = np.asarray(y_score, dtype=float)
    if score.ndim != 1:
        score = score.reshape(-1)
    return np.nan_to_num(score, nan=0.5, posinf=1.0, neginf=0.0)
