from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PatchClassifier:
    def __init__(
        self,
        class_names: list[str],
        model_type: str = 'ensemble',
        random_state: int = 42,
        max_iter: int = 1200,
    ) -> None:
        self.class_names = class_names
        self.model_type = model_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = self._build_model()

    def _build_model(self) -> BaseEstimator:
        if self.model_type == 'logreg':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=self.max_iter, class_weight='balanced', random_state=self.random_state)),
            ])
        if self.model_type == 'mlp':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation='relu',
                    solver='adam',
                    alpha=1e-4,
                    batch_size=128,
                    learning_rate_init=1e-3,
                    max_iter=self.max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    random_state=self.random_state,
                )),
            ])
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=max(300, self.max_iter // 2),
                class_weight='balanced_subsample',
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=self.random_state,
            )
        if self.model_type == 'extratrees':
            return ExtraTreesClassifier(
                n_estimators=max(400, self.max_iter // 2),
                class_weight='balanced',
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=self.random_state,
            )
        if self.model_type == 'ensemble':
            lr = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=self.max_iter, class_weight='balanced', random_state=self.random_state)),
            ])
            mlp = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation='relu',
                    solver='adam',
                    alpha=1e-4,
                    batch_size=128,
                    learning_rate_init=8e-4,
                    max_iter=self.max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=35,
                    random_state=self.random_state,
                )),
            ])
            et = ExtraTreesClassifier(
                n_estimators=max(500, self.max_iter // 2),
                class_weight='balanced',
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=self.random_state,
            )
            return VotingClassifier(
                estimators=[('lr', lr), ('mlp', mlp), ('et', et)],
                voting='soft',
                n_jobs=-1,
            )
        raise ValueError(f'Unsupported model_type: {self.model_type}')

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        probas = self.predict_proba(x)
        preds = self.predict(x)
        malignant_index = list(self.model.classes_).index('malignant') if 'malignant' in self.model.classes_ else -1
        score = probas[:, malignant_index] if malignant_index >= 0 else probas.max(axis=1)

        from breastpath_copilot.metrics import compute_binary_metrics

        metrics = compute_binary_metrics(y_true=y, y_pred=preds, y_score=score)
        metrics['model_type'] = self.model_type
        metrics['max_iter'] = int(self.max_iter)
        return metrics

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'class_names': self.class_names,
            'model_type': self.model_type,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> 'PatchClassifier':
        payload = joblib.load(path)
        obj = cls(
            class_names=payload['class_names'],
            model_type=payload.get('model_type', 'ensemble'),
            random_state=payload.get('random_state', 42),
            max_iter=payload.get('max_iter', 1200),
        )
        obj.model = payload['model']
        return obj
