"""Processing time estimates, learned from this PC's own past runs.

Every finished step of an automatic run is recorded (level, number of
photos, seconds, hardware) in a small JSON history in the user's data
folder. An estimate for a new run uses the median seconds per photo of
the same step and level measured on this same hardware; until a step has
history here, it falls back to the times measured on the reference PC
(Ryzen 7 5700G + RTX 3060 Ti, 56 photos of 20 MP), labelled as such.

Every step scales roughly linearly with the number of photos: alignment
matches each photo against a fixed number of neighbors, the dense cloud
works per photo at a fixed image size, and the orthomosaic area grows
with the photos.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HISTORY_MAX_ENTRIES = 300

# Seconds per photo on the reference PC (56-photo fazenda 20046 flight).
REFERENCE_SECONDS_PER_PHOTO: dict[str, dict[str, float]] = {
    "baixo": {
        "importar": 0.2,
        "alinhar": 2.5,
        "ajustar": 1.5,
        "densa": 4.5,
        "dem": 0.4,
        "ortho": 1.0,
        "relatorio": 0.6,
    },
    "medio": {
        "importar": 0.2,
        "alinhar": 5.4,
        "ajustar": 2.7,
        "densa": 36.0,
        "dem": 1.1,
        "ortho": 3.2,
        "relatorio": 0.8,
    },
    "alto": {
        "importar": 0.2,
        "alinhar": 5.4,
        "ajustar": 2.7,
        "densa": 145.0,
        "dem": 3.0,
        "ortho": 3.2,
        "relatorio": 1.0,
    },
}


def history_path() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "htrmapper" / "tempos_processamento.json"


def hardware_signature() -> str:
    from htrmapper.core.system_info import detect_system_info

    info = detect_system_info()
    gpu = info.gpus[0].name if info.gpus else "sem GPU NVIDIA"
    return f"{info.cpu_name} | {gpu}"


def _load(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def record_step(
    step: str, level: str, num_photos: int, seconds: float, hardware: str, path: Path | None = None
) -> None:
    """Append one measured step to the history (never raises: losing a
    timing sample must not fail a processing run)."""
    if num_photos <= 0 or seconds <= 0:
        return
    path = path or history_path()
    entries = _load(path)
    entries.append(
        {
            "step": step,
            "level": level,
            "num_photos": num_photos,
            "seconds": round(seconds, 1),
            "hardware": hardware,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries[-HISTORY_MAX_ENTRIES:], indent=1), encoding="utf-8")
    except OSError:
        pass


@dataclass
class StepEstimate:
    seconds: float
    from_history: bool  # False: reference-PC figure, not yet measured here


def estimate(
    level: str, num_photos: int, hardware: str, steps: list[str], path: Path | None = None
) -> dict[str, StepEstimate]:
    entries = [e for e in _load(path or history_path()) if e.get("hardware") == hardware and e.get("level") == level]
    result = {}
    for step in steps:
        rates = [e["seconds"] / e["num_photos"] for e in entries if e.get("step") == step and e.get("num_photos")]
        if rates:
            result[step] = StepEstimate(statistics.median(rates[-10:]) * num_photos, True)
        else:
            result[step] = StepEstimate(REFERENCE_SECONDS_PER_PHOTO[level][step] * num_photos, False)
    return result


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min" if minutes >= 10 else f"{minutes} min {secs:02d} s"
    return f"{secs} s"
