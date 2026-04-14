## Phase 2 Report

So far, we have built a working AI tool that can analyze engine audio recordings and classify whether the engine is operating normally or abnormally with a relatively high level of accuracy.

The model uses audio data from public datasets and processes short clips to generate predictions, and we have a basic interface that allows users to input audio and receive results. It is trained using examples from a wide variety of failure points such as batteries, belts, ignition components, and pumps. To ensure it can recognize healthy engines, multiple normal engine audios were anylized at idle and startup. By testing it on other vehicles, we demonstrate that our core pipeline data processing, model training, and inference are functioning as intended.

However, the current system has some limitations. It only performs binary classification and does not identify specific engine issues; we have observed some inconsistencies due to variations in audio quality and background noise. It only supports .wav audio files, which may be a limitation for some users.

In Phase 3, we plan to improve the model’s reliability by expanding the training data and reduce errors in audio input. We can increase the value of the output by providing more detailed diagnostic information beyond simple healthy-versus-abnormal classification. Another possible improvement is to expand the range of supported file types for user input, such as .mp3, and .m4a.
