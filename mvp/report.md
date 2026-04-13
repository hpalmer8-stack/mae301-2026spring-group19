## Phase 2 Report

So far, we have built a working AI tool that can analyze engine audio recordings and classify whether the engine is operating normally or abnormally with a relatively high level of accuracy.

The model uses audio data from public datasets and processes short clips to generate predictions, and we have a basic interface that allows users to input audio and receive results.

This demonstrates that our core pipeline data processing, model training, and inference are functioning as intended.

However, the current system has some limitations.

It only performs binary classification and does not identify specific engine issues; we have observed some inconsistencies due to variations in audio quality and background noise.

In Phase 3, we plan to improve the model’s reliability, reduce errors in audio input, and expand the system to provide more detailed diagnostic information beyond simple healthy-versus-abnormal classification.
