from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from breastpath_copilot.core.fm_cache import FMEmbeddingCache
from breastpath_copilot.core.fm_embeddings import (
    BACKEND_REGISTRY,
    FMBackend,
    build_backend,
)
from breastpath_copilot.data_cohorts import (
    BACH_DIR_TO_LABEL,
    BRACS_DIR_TO_LABEL,
    leave_one_cohort_out_splits,
    parse_bach_image_path,
    parse_bracs_image_path,
)
from breastpath_copilot.schemas import FMEmbeddingMetadata


class _DummyBackend(FMBackend):
    """Test backend that fakes inference without loading any model weights."""

    name = "dummy"
    model_id = "dummy/test"
    embedding_dim = 8
    input_size = 32
    requires_hf_token = False

    def _load_model(self):  # pragma: no cover - bypassed
        raise NotImplementedError

    def _forward(self, batch):  # pragma: no cover - bypassed
        raise NotImplementedError

    def encode_images(self, images):
        images = list(images)
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.random.default_rng(0).standard_normal((len(images), self.embedding_dim)).astype(np.float32)


def test_backend_registry_has_three_pathology_fms():
    assert {"uni", "virchow2", "phikon_v2"} <= set(BACKEND_REGISTRY)
    assert BACKEND_REGISTRY["uni"].embedding_dim == 1024
    assert BACKEND_REGISTRY["virchow2"].embedding_dim == 2560
    assert BACKEND_REGISTRY["phikon_v2"].embedding_dim == 1024


def test_build_backend_unknown_name_raises():
    with pytest.raises(KeyError):
        build_backend("not_a_real_backend")


def test_cache_round_trip(tmp_path: Path):
    cache = FMEmbeddingCache(tmp_path, cohort="breakhis", fm_name="dummy")
    assert not cache.exists()

    df = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "image_path": ["/x", "/y", "/z"],
            "label": ["benign", "invasive", "benign"],
        }
    )
    vectors = np.random.default_rng(1).standard_normal((3, 8)).astype(np.float32)
    metadata = cache.write(df, vectors, model_id="dummy/test", embedding_dim=8)
    assert isinstance(metadata, FMEmbeddingMetadata)
    assert metadata.num_records == 3
    assert cache.exists()

    loaded = cache.load_embeddings(mmap=False)
    np.testing.assert_allclose(loaded, vectors)


def test_cache_append_deduplicates_on_image_id(tmp_path: Path):
    cache = FMEmbeddingCache(tmp_path, cohort="breakhis", fm_name="dummy")
    df1 = pd.DataFrame({"image_id": ["a", "b"], "image_path": ["/x", "/y"]})
    cache.write(df1, np.zeros((2, 4), dtype=np.float32), model_id="dummy/test", embedding_dim=4)

    df2 = pd.DataFrame({"image_id": ["b", "c", "d"], "image_path": ["/y", "/z", "/w"]})
    metadata = cache.write(
        df2, np.ones((3, 4), dtype=np.float32), model_id="dummy/test", embedding_dim=4
    )
    assert metadata.num_records == 4  # a, b, c, d (b deduped)

    full_manifest = cache.load_manifest()
    assert list(full_manifest["image_id"]) == ["a", "b", "c", "d"]
    assert list(full_manifest["row_idx"]) == [0, 1, 2, 3]


def test_cache_missing_rows(tmp_path: Path):
    cache = FMEmbeddingCache(tmp_path, cohort="breakhis", fm_name="dummy")
    df = pd.DataFrame({"image_id": ["a", "b"], "image_path": ["/x", "/y"]})
    cache.write(df, np.zeros((2, 4), dtype=np.float32), model_id="dummy/test", embedding_dim=4)

    probe = pd.DataFrame({"image_id": ["a", "c", "d"], "image_path": ["/x", "/zz", "/yy"]})
    missing = cache.missing_rows(probe)
    assert list(missing["image_id"]) == ["c", "d"]


def test_cache_slice_by_image_ids_preserves_request_order(tmp_path: Path):
    cache = FMEmbeddingCache(tmp_path, cohort="breakhis", fm_name="dummy")
    df = pd.DataFrame({"image_id": ["a", "b", "c"], "image_path": ["/x", "/y", "/z"]})
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache.write(df, vectors, model_id="dummy/test", embedding_dim=4)

    rows, sliced = cache.slice_by_image_ids(["c", "a", "missing"])
    assert list(rows["image_id"]) == ["c", "a"]
    np.testing.assert_allclose(sliced[0], vectors[2])
    np.testing.assert_allclose(sliced[1], vectors[0])


def test_dummy_backend_writes_through_cache(tmp_path: Path):
    """End-to-end: a manifest + a dummy backend produces a valid cache."""
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    paths = []
    for i, name in enumerate(["a", "b", "c"]):
        p = images_dir / f"{name}.png"
        Image.new("RGB", (16, 16), color=(i * 50, 10, 200 - i * 30)).save(p)
        paths.append(str(p))

    df = pd.DataFrame({"image_id": ["a", "b", "c"], "image_path": paths})
    backend = _DummyBackend()
    vectors = backend.encode_paths(df["image_path"])
    assert vectors.shape == (3, 8)

    cache = FMEmbeddingCache(tmp_path / "cache", cohort="breakhis", fm_name=backend.name)
    metadata = cache.write(
        df,
        vectors,
        model_id=backend.model_id,
        embedding_dim=backend.embedding_dim,
        input_size=backend.input_size,
    )
    assert metadata.num_records == 3
    assert metadata.fm_name == "dummy"


# --------------------------------------------------------------------------- #
# Cohort parsers
# --------------------------------------------------------------------------- #

def test_bach_parser_maps_insitu_to_dcis():
    info = parse_bach_image_path(Path("/data/bach/Photos/InSitu/is042.tif"))
    assert info.label == "dcis"
    assert info.subtype == "in_situ_carcinoma"


def test_bach_parser_handles_case_variations():
    info = parse_bach_image_path(Path("/data/bach/Photos/in_situ/case01.png"))
    assert info.label == "dcis"


@pytest.mark.parametrize(
    "folder,expected_label",
    [
        # Numbered prefix as used in the official BRACS_RoI release.
        ("0_N", "normal"),
        ("1_PB", "benign"),
        ("2_UDH", "benign"),
        ("3_FEA", "atypia"),
        ("4_ADH", "atypia"),
        ("5_DCIS", "dcis"),
        ("6_IC", "invasive"),
        # Bare abbreviations and spelled-out variants are also accepted.
        ("Normal", "normal"),
        ("Pathological_Benign", "benign"),
        ("UDH", "benign"),
        ("FEA", "atypia"),
        ("ADH", "atypia"),
        ("DCIS", "dcis"),
        ("IC", "invasive"),
    ],
)
def test_bracs_parser_maps_early_detection_classes(folder: str, expected_label: str):
    info = parse_bracs_image_path(Path(f"/data/bracs/RoI/{folder}/BRACS_017_RoI_1.png"))
    assert info.label == expected_label
    assert info.patient_id == "bracs_017"


def test_bracs_parser_extracts_split_when_path_contains_train_val_test():
    paths = [
        ("/data/BRACS/BRACS_RoI/latest_version/train/4_ADH/BRACS_1238_ADH_1.png", "train"),
        ("/data/BRACS/BRACS_RoI/latest_version/test/5_DCIS/BRACS_1003697_DCIS_1.png", "test"),
        ("/data/BRACS/BRACS_RoI/latest_version/val/0_N/BRACS_1228_N_3.png", "val"),
        # validation -> normalised to val
        ("/data/bracs/Images/validation/3_FEA/BRACS_1003693_FEA_1.png", "val"),
        # no split in path
        ("/data/bracs/RoI/5_DCIS/BRACS_2222_DCIS_4.png", None),
    ]
    for raw, expected_split in paths:
        info = parse_bracs_image_path(Path(raw))
        assert info.split == expected_split, f"{raw}: got {info.split!r}, want {expected_split!r}"


def test_build_bracs_manifest_defaults_to_latest_and_avoids_duplicates(tmp_path: Path):
    """Synthetic mirror of the real BRACS_RoI layout. Walking the whole tree
    naively would double-count the same image between latest_version and
    previous_versions; with ``version='latest'`` (default), only the new
    release should be included."""
    from PIL import Image as PILImage

    from breastpath_copilot.data_cohorts import build_bracs_manifest

    root = tmp_path / "BRACS_RoI"
    # latest_version
    latest_adh = root / "latest_version" / "train" / "4_ADH"
    latest_adh.mkdir(parents=True)
    PILImage.new("RGB", (16, 16)).save(latest_adh / "BRACS_1238_ADH_1.png")
    PILImage.new("RGB", (16, 16)).save(latest_adh / "BRACS_1238_ADH_2.png")
    latest_dcis_test = root / "latest_version" / "test" / "5_DCIS"
    latest_dcis_test.mkdir(parents=True)
    PILImage.new("RGB", (16, 16)).save(latest_dcis_test / "BRACS_1004_DCIS_1.png")
    # previous_versions: same image_ids that would double-count if walked
    v1_adh = root / "previous_versions" / "Version1_MedIA" / "Images" / "train" / "4_ADH"
    v1_adh.mkdir(parents=True)
    PILImage.new("RGB", (16, 16)).save(v1_adh / "BRACS_1238_ADH_1.png")

    df = build_bracs_manifest(root)  # version='latest' is default
    assert len(df) == 3, f"expected 3 rows, got {len(df)}"
    assert set(df["split"]) == {"train", "test"}
    # Patient id extracted from filename, not folder.
    assert "bracs_1238" in set(df["patient_id"])
    # Sanity: no rows reference the previous_versions subtree.
    assert not df["image_path"].str.contains("previous_versions").any()

    # Opting in to 'all' should produce both subtrees (used for audits, not training).
    df_all = build_bracs_manifest(root, version="all")
    assert len(df_all) == 4


def test_leave_one_cohort_out_yields_one_split_per_cohort():
    df = pd.DataFrame(
        {
            "cohort": ["breakhis"] * 3 + ["bach"] * 2 + ["bracs"] * 4,
            "image_id": [f"i{i}" for i in range(9)],
        }
    )
    splits = list(leave_one_cohort_out_splits(df))
    held_out = sorted(name for name, _, _ in splits)
    assert held_out == ["bach", "bracs", "breakhis"]
    for held, train, test in splits:
        assert not (train["cohort"].astype(str) == held).any()
        assert (test["cohort"].astype(str) == held).all()


def test_label_maps_cover_expected_vocabulary():
    # BACH covers normal/benign/dcis/invasive.
    assert set(BACH_DIR_TO_LABEL.values()) == {"normal", "benign", "dcis", "invasive"}
    # BRACS spans the full unified vocabulary including atypia.
    assert {"normal", "benign", "atypia", "dcis", "invasive"} <= set(BRACS_DIR_TO_LABEL.values())
