from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from engine_health_ai.model import score_audio_file


SPACE_ROOT = Path(__file__).resolve().parent
MODEL_PATH = SPACE_ROOT / "artifacts" / "engine_health_model.joblib"
EXAMPLES_DIR = SPACE_ROOT / "examples"


def predict_engine_health(audio_file: str | None) -> tuple[str, str]:
    if not audio_file:
        return "No audio file provided.", "{}"

    result = score_audio_file(MODEL_PATH, audio_file)
    summary = (
        f"Prediction: {result['prediction']}\n"
        f"Healthy confidence: {result['healthy_confidence']:.2%}\n"
        f"Unhealthy confidence: {result['unhealthy_confidence']:.2%}\n"
        f"Recommendation: {result['recommendation']}"
    )
    return summary, json.dumps(result, indent=2)


def example_paths() -> list[list[str]]:
    if not EXAMPLES_DIR.exists():
        return []
    return [[str(path)] for path in sorted(EXAMPLES_DIR.glob("*.wav"))]


demo = gr.Interface(
    fn=predict_engine_health,
    inputs=gr.Audio(type="filepath", label="Upload a 5-second .wav engine clip"),
    outputs=[
        gr.Textbox(label="Prediction Summary", lines=5),
        gr.Code(label="Detailed JSON Output", language="json"),
    ],
    title="Engine Health Audio Classifier",
    description=(
        "Upload a short engine audio clip to classify it as healthy or unhealthy. "
        "This demo uses a trained binary audio classifier built for the MAE 301 project."
    ),
    examples=example_paths(),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
