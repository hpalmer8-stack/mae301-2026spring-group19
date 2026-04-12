---
title: Engine Health Audio Classifier
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.25.2
python_version: 3.11
app_file: app.py
pinned: false
---

# Engine Health Audio Classifier

This Hugging Face Space hosts a browser demo for the MAE 301 engine health MVP. Users can upload a short `.wav` engine clip and receive a `healthy` or `unhealthy` prediction with confidence scores and a simple recommendation.

## Included assets

- Trained model artifact in `artifacts/engine_health_model.joblib`
- Python inference package in `engine_health_ai/`
- Example `.wav` clips in `examples/`

## Notes

- The app performs binary classification only: `healthy` or `unhealthy`.
- The uploaded file should be a short `.wav` engine recording.
- Python is pinned to `3.11` because local Python `3.14` currently causes a `pydantic-core` build issue when installing Gradio on Windows; Hugging Face Spaces supports explicitly setting the Python version in the README YAML. Source: [Spaces Configuration Reference](https://huggingface.co/docs/hub/main/spaces-config-reference)
