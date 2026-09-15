"""Fase 7: background execution for long-running pipeline stages.

Every Fase 2-6 pipeline call (`run_structure_from_motion`,
`run_gnss_weighted_bundle_adjustment`, `run_dense_reconstruction`,
`run_dem_generation`, `run_orthomosaic_generation`) blocks the calling
thread for real work -- seconds on the synthetic test scenes, potentially
hours on a real multi-hundred-image flight. Calling any of them directly
from Qt's main thread would freeze the whole GUI (tree, buttons, system
monitor) for that entire time; `PipelineWorker` moves one such call onto a
`QThread` and reports back only via Qt signals, which Qt marshals safely
across threads -- the worker thread itself never touches a widget.

Cancellation and progress are opt-in per function, detected by inspecting
its signature rather than assumed: `run_gnss_weighted_bundle_adjustment`
(Fase 3) has no native cancellation hook in this pycolmap version (Ceres'
`solve()` is a single opaque call), so it simply does not declare a
`cancellation_token` parameter, and this worker then never manufactures
one -- the GUI reads `worker.supports_cancellation` to decide whether to
even show a Cancelar button for that stage, rather than showing one that
silently does nothing.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pycolmap
from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    # A phase label (coarse, COLMAP-backed stages) or a real (done, total)
    # count (Fase 6's own per-camera loop) -- never both on one signal, so
    # each carries an unambiguous, real value, never a fabricated percentage.
    phase_changed = Signal(str)
    progress_changed = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(self, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        params = inspect.signature(func).parameters
        self.supports_cancellation = "cancellation_token" in params
        self.supports_progress = "progress_callback" in params
        self.cancellation_token = pycolmap.CancellationToken() if self.supports_cancellation else None

    def cancel(self) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.cancel()

    def _on_progress(self, *args: Any) -> None:
        if len(args) == 1:
            self.phase_changed.emit(args[0])
        else:
            done, total = args
            self.progress_changed.emit(done, total)

    def run(self) -> None:
        kwargs = dict(self._kwargs)
        if self.supports_cancellation:
            kwargs["cancellation_token"] = self.cancellation_token
        if self.supports_progress:
            kwargs["progress_callback"] = self._on_progress

        try:
            result = self._func(*self._args, **kwargs)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 -- surfaced to the GUI, never swallowed
            self.failed.emit(exc)
        else:
            self.finished_ok.emit(result)
