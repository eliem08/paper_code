from __future__ import annotations

from collections import defaultdict

from breastpath_copilot.schemas import PatchPrediction, SlidePrediction


def aggregate_patch_predictions(
    slide_id: str,
    patch_predictions: list[PatchPrediction],
    class_names: list[str],
) -> SlidePrediction:
    if not patch_predictions:
        uniform = {c: 1.0 / len(class_names) for c in class_names}
        return SlidePrediction(slide_id=slide_id, predicted_label=class_names[0], probabilities=uniform)

    summed = defaultdict(float)
    suspicious_patch_ids: list[str] = []

    for pred in patch_predictions:
        for label, prob in pred.probabilities.items():
            summed[label] += prob
        if pred.predicted_label in {"atypia", "dcis", "invasive"}:
            suspicious_patch_ids.append(pred.patch_id)

    total = sum(summed.values()) or 1.0
    probs = {label: summed[label] / total for label in class_names}
    predicted_label = max(probs, key=probs.get)

    return SlidePrediction(
        slide_id=slide_id,
        predicted_label=predicted_label,
        probabilities=probs,
        suspicious_patch_ids=suspicious_patch_ids[:10],
    )
