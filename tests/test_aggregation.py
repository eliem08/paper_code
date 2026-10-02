from breastpath_copilot.core.aggregation import aggregate_patch_predictions
from breastpath_copilot.schemas import PatchPrediction


def test_aggregate_patch_predictions() -> None:
    preds = [
        PatchPrediction(
            slide_id="s1",
            patch_id="p1",
            predicted_label="dcis",
            probabilities={"normal": 0.1, "benign": 0.1, "atypia": 0.2, "dcis": 0.5, "invasive": 0.1},
        ),
        PatchPrediction(
            slide_id="s1",
            patch_id="p2",
            predicted_label="invasive",
            probabilities={"normal": 0.05, "benign": 0.05, "atypia": 0.1, "dcis": 0.2, "invasive": 0.6},
        ),
    ]
    result = aggregate_patch_predictions("s1", preds, ["normal", "benign", "atypia", "dcis", "invasive"])
    assert result.predicted_label in {"dcis", "invasive"}
    assert len(result.suspicious_patch_ids) == 2
