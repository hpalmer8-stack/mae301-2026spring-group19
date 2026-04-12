from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


NORMAL_PREFIXES = ("normal_", "healthy")


@dataclass(frozen=True)
class AudioExample:
    path: Path
    state: str
    condition: str
    label: int


def is_healthy_condition(condition_name: str) -> bool:
    normalized = condition_name.strip().lower()
    return normalized.startswith(NORMAL_PREFIXES)


def discover_dataset(dataset_root: str | Path) -> list[AudioExample]:
    root = Path(dataset_root)
    examples: list[AudioExample] = []

    for state_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for condition_dir in sorted(path for path in state_dir.iterdir() if path.is_dir()):
            label = 1 if is_healthy_condition(condition_dir.name) else 0
            for audio_path in sorted(condition_dir.glob("*.wav")):
                examples.append(
                    AudioExample(
                        path=audio_path,
                        state=state_dir.name,
                        condition=condition_dir.name,
                        label=label,
                    )
                )

    if not examples:
        raise FileNotFoundError(f"No .wav files were found under {root}")

    return examples


def summarize_dataset(examples: list[AudioExample]) -> dict[str, object]:
    healthy = sum(example.label for example in examples)
    unhealthy = len(examples) - healthy
    by_state = Counter(example.state for example in examples)
    by_condition = Counter(example.condition for example in examples)
    return {
        "total_examples": len(examples),
        "healthy_examples": healthy,
        "unhealthy_examples": unhealthy,
        "states": dict(by_state),
        "conditions": dict(by_condition),
    }
