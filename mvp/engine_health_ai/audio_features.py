from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
CLIP_DURATION_SECONDS = 5
TARGET_NUM_SAMPLES = TARGET_SAMPLE_RATE * CLIP_DURATION_SECONDS


def load_audio(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    sample_rate, samples = wavfile.read(str(path))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    samples = samples.astype(np.float32)
    max_abs = np.max(np.abs(samples)) if samples.size else 0.0
    if max_abs > 0:
        samples /= max_abs

    if sample_rate != target_sample_rate:
        samples = resample_poly(samples, target_sample_rate, sample_rate)

    return fix_length(samples, TARGET_NUM_SAMPLES)


def fix_length(samples: np.ndarray, target_num_samples: int) -> np.ndarray:
    if samples.size >= target_num_samples:
        return samples[:target_num_samples]

    padded = np.zeros(target_num_samples, dtype=np.float32)
    padded[: samples.size] = samples
    return padded


def _frame_signal(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if samples.size < frame_size:
        samples = fix_length(samples, frame_size)

    num_frames = 1 + max(0, (samples.size - frame_size) // hop_size)
    frames = np.zeros((num_frames, frame_size), dtype=np.float32)
    for idx in range(num_frames):
        start = idx * hop_size
        frames[idx] = samples[start : start + frame_size]
    return frames


def extract_features(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    frame_size = int(sample_rate * 0.05)
    hop_size = int(sample_rate * 0.025)
    frames = _frame_signal(samples, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window

    rms = np.sqrt(np.mean(windowed**2, axis=1) + 1e-8)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-8
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    spectral_sum = np.sum(spectrum, axis=1)
    centroid = np.sum(spectrum * freqs, axis=1) / spectral_sum
    bandwidth = np.sqrt(np.sum(((freqs - centroid[:, None]) ** 2) * spectrum, axis=1) / spectral_sum)

    cumulative = np.cumsum(spectrum, axis=1)
    rolloff_threshold = 0.85 * spectral_sum
    rolloff = freqs[np.argmax(cumulative >= rolloff_threshold[:, None], axis=1)]
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / np.mean(spectrum, axis=1)

    normalized_spectrum = spectrum / spectral_sum[:, None]
    spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum), axis=1)
    dominant_freq = freqs[np.argmax(spectrum, axis=1)]

    autocorr = np.correlate(samples, samples, mode="full")[samples.size - 1 :]
    autocorr = autocorr[: sample_rate]
    autocorr[0] = 0
    peak_lag = int(np.argmax(autocorr)) if autocorr.size else 0
    periodicity_hz = sample_rate / peak_lag if peak_lag > 0 else 0.0

    band_edges = [0, 200, 500, 1000, 2000, 4000, 8000]
    band_energies: list[float] = []
    for start_hz, end_hz in zip(band_edges[:-1], band_edges[1:]):
        band_mask = (freqs >= start_hz) & (freqs < end_hz)
        band_power = np.mean(spectrum[:, band_mask], axis=1)
        band_energies.extend(
            [
                float(np.mean(band_power)),
                float(np.std(band_power)),
            ]
        )

    crest_factor = np.max(np.abs(samples)) / (np.sqrt(np.mean(samples**2)) + 1e-8)
    dynamic_range = np.percentile(np.abs(samples), 95) - np.percentile(np.abs(samples), 5)

    features = [
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(bandwidth)),
        float(np.std(bandwidth)),
        float(np.mean(rolloff)),
        float(np.std(rolloff)),
        float(np.mean(flatness)),
        float(np.std(flatness)),
        float(np.mean(spectral_entropy)),
        float(np.std(spectral_entropy)),
        float(np.mean(dominant_freq)),
        float(np.std(dominant_freq)),
        float(crest_factor),
        float(dynamic_range),
        float(periodicity_hz),
        float(np.mean(np.abs(samples))),
        float(np.std(samples)),
        float(np.max(np.abs(samples))),
        *band_energies,
    ]

    cleaned = [0.0 if math.isnan(value) or math.isinf(value) else value for value in features]
    return np.asarray(cleaned, dtype=np.float32)


def extract_features_from_path(path: str | Path) -> np.ndarray:
    samples = load_audio(path)
    return extract_features(samples)
