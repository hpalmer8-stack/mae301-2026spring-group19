from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .audio_features import extract_features_from_path
from .data import AudioExample


FEATURE_NAMES = [
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "rolloff_std",
    "flatness_mean",
    "flatness_std",
    "entropy_mean",
    "entropy_std",
    "dominant_freq_mean",
    "dominant_freq_std",
    "crest_factor",
    "dynamic_range",
    "periodicity_hz",
    "abs_mean",
    "signal_std",
    "peak_amplitude",
    "band_0_200_mean",
    "band_0_200_std",
    "band_200_500_mean",
    "band_200_500_std",
    "band_500_1000_mean",
    "band_500_1000_std",
    "band_1000_2000_mean",
    "band_1000_2000_std",
    "band_2000_4000_mean",
    "band_2000_4000_std",
    "band_4000_8000_mean",
    "band_4000_8000_std",
]


@dataclass
class TrainArtifacts:
    pipeline: Pipeline
    healthy_feature_mean: np.ndarray
    feature_names: list[str]


def build_feature_matrix(examples: list[AudioExample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([extract_features_from_path(example.path) for example in examples])
    y = np.asarray([example.label for example in examples], dtype=np.int32)
    return x, y


def train_model(examples: list[AudioExample], random_state: int = 42) -> tuple[TrainArtifacts, dict[str, object]]:
    x, y = build_feature_matrix(examples)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    train_probs = pipeline.predict_proba(x_train)[:, 1]
    train_pred = (train_probs >= 0.5).astype(np.int32)
    test_probs = pipeline.predict_proba(x_test)[:, 1]
    test_pred = (test_probs >= 0.5).astype(np.int32)

    healthy_feature_mean = x_train[y_train == 1].mean(axis=0)
    artifacts = TrainArtifacts(
        pipeline=pipeline,
        healthy_feature_mean=healthy_feature_mean,
        feature_names=FEATURE_NAMES,
    )

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            test_pred,
            target_names=["unhealthy", "healthy"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return artifacts, metrics


def save_artifacts(artifacts: TrainArtifacts, destination: str | Path) -> None:
    payload = {
        "pipeline": artifacts.pipeline,
        "healthy_feature_mean": artifacts.healthy_feature_mean,
        "feature_names": artifacts.feature_names,
    }
    joblib.dump(payload, destination)


def load_artifacts(source: str | Path) -> TrainArtifacts:
    payload = joblib.load(source)
    return TrainArtifacts(
        pipeline=payload["pipeline"],
        healthy_feature_mean=payload["healthy_feature_mean"],
        feature_names=payload["feature_names"],
    )


def score_audio_file(model_path: str | Path, audio_path: str | Path) -> dict[str, object]:
    artifacts = load_artifacts(model_path)
    features = extract_features_from_path(audio_path).reshape(1, -1)
    healthy_probability = float(artifacts.pipeline.predict_proba(features)[0, 1])
    unhealthy_probability = 1.0 - healthy_probability
    predicted_label = "healthy" if healthy_probability >= 0.5 else "unhealthy"

    delta = np.abs(features[0] - artifacts.healthy_feature_mean)
    top_indices = np.argsort(delta)[-3:][::-1]
    explanation = [
        {
            "feature": artifacts.feature_names[index],
            "distance_from_healthy_baseline": float(delta[index]),
        }
        for index in top_indices
    ]

    recommendation = (
        "Engine sound appears normal. Continue routine monitoring."
        if predicted_label == "healthy"
        else "Engine sound appears abnormal. Schedule a mechanical inspection soon."
    )
    summary_text = (
        f"The clip was classified as {predicted_label} with "
        f"{max(healthy_probability, unhealthy_probability):.1%} confidence. "
        f"{recommendation}"
    )

    return {
        "prediction": predicted_label,
        "healthy_confidence": round(healthy_probability, 4),
        "unhealthy_confidence": round(unhealthy_probability, 4),
        "summary_text": summary_text,
        "recommendation": recommendation,
        "top_audio_deviations": explanation,
    }
