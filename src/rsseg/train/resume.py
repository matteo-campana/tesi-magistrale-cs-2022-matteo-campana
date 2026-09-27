# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pause and exact resume of segmentation training and DINO pre-training.

A resume checkpoint (``<run>/resume/``, see :mod:`rsseg.runstate`) holds everything an
uninterrupted run would still have in memory:

* all model variables, including the state of the dropout / stochastic-depth seed generators
  and, for DINO, the EMA teacher and the loss center;
* the optimizer variables (AdamW moments, iteration counter — which drives the DINO schedules —
  and the gradient-accumulation buffers);
* mid-epoch, the metric accumulators (running loss, confusion matrix);
* the position in the training stream (:class:`rsseg.data.datasets.StreamSampler`): order of the
  crops and their augmentation depend only on it;
* the state of the callbacks (early stopping, top-k checkpoints, progress, MLflow step) and the
  MLflow run id, so the resumed run continues the same MLflow run.

:class:`ResumeCheckpoint` writes it when a pause is requested (:mod:`rsseg.control`) and every
``checkpoint_every_min`` minutes (recovery after a crash or a power loss). A pause requested
during validation takes effect at the end of the epoch. :func:`fit_resumable` runs ``fit`` from
the saved position; a run resumed mid-epoch first completes that epoch with a short ``fit``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from pathlib import Path

import keras
import numpy as np

from rsseg import control, progress
from rsseg.runstate import STATE_ARRAYS, STATE_JSON, environment, read_state, resume_dir

log = logging.getLogger(__name__)


def _arrays(variables) -> list[np.ndarray]:
    return [np.asarray(keras.ops.convert_to_numpy(v)) for v in variables]


def _assign(variables, arrays: list[np.ndarray], what: str) -> None:
    variables = list(variables)
    if len(variables) != len(arrays):
        raise ValueError(
            f"resume checkpoint does not match the {what}: {len(arrays)} saved variables, "
            f"{len(variables)} in the model (was the configuration or the code changed?)"
        )
    for v, a in zip(variables, arrays, strict=True):
        if tuple(v.shape) != a.shape:
            raise ValueError(
                f"resume checkpoint: shape {a.shape} saved for {v.path}, {v.shape} now"
            )
        v.assign(a)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def gpu_name() -> str | None:
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def save_phase(run_dir: str | Path, meta: dict) -> None:
    """Write a resume ``state.json`` without arrays (e.g. a run paused during the evaluation)."""
    d = resume_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / STATE_ARRAYS).unlink(missing_ok=True)  # the arrays of the training are not needed
    _write_json(d / STATE_JSON, {**meta, "saved": time.time(),
                                 "env": {**environment(), "gpu": gpu_name()}})  # fmt: skip


def discard(run_dir: str | Path) -> None:
    """Delete the resume checkpoint (the run finished, or it was stopped)."""
    shutil.rmtree(resume_dir(run_dir), ignore_errors=True)


def load(run_dir: str | Path) -> dict:
    """Metadata and arrays of the resume checkpoint of ``run_dir``."""
    state = read_state(run_dir)
    if state is None:
        raise FileNotFoundError(f"no resume checkpoint in {resume_dir(run_dir)}")
    if state.get("phase") == "train":
        with np.load(resume_dir(run_dir) / STATE_ARRAYS) as z:
            if int(z["generation"]) != state["generation"]:
                raise ValueError(f"{resume_dir(run_dir)}: state.json and state.npz do not match")
            state["arrays"] = {
                kind: [z[f"{kind}{i}"] for i in range(int(z[f"n_{kind}"]))]
                for kind in ("model", "optimizer", "metrics")
            }
    return state


def _truncate_csv(path: Path, from_epoch: int) -> None:
    """Drop the rows of epochs ``>= from_epoch`` (0-based) written after the checkpoint."""
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    kept = [rows[0]] + [r for r in rows[1:] if r and int(float(r[0])) < from_epoch]
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(kept)


class ResumeCheckpoint(keras.callbacks.Callback):
    """Advance the training stream, obey pause/stop requests and write resume checkpoints.

    Put it **last** in the callback list: it restores the state of the other callbacks after
    their ``on_train_begin`` (which resets e.g. the early stopping) and saves their state after
    their ``on_epoch_end``.

    ``callbacks`` maps a name to a callback whose state is saved: callbacks with
    ``get_state``/``set_state``, ``EarlyStopping`` and ``MLflowMetricsLogger`` are supported.
    """

    def __init__(self, run_dir: str | Path, kind: str, sampler, epochs: int, steps: int,
                 every_min: float = 0.0, callbacks: dict | None = None,
                 state: dict | None = None):  # fmt: skip
        super().__init__()
        self.run_dir = Path(run_dir)
        self.kind = kind
        self.sampler = sampler
        self.epochs, self.steps = epochs, steps
        self.every_s = every_min * 60.0
        self.tracked = callbacks or {}
        self.state = state
        self.run_id: str | None = state.get("run_id") if state else None
        self.epoch = state["epoch"] if state else 0
        self.batch = state["batch"] if state else 0
        self.active_before = state.get("active_s", 0.0) if state else 0.0
        self.t_start = time.time()
        self.last_save = time.time()
        self.generation = state["generation"] if state else 0
        self._restore_all = state is not None
        self._restore_metrics = bool(state and state["batch"])
        self._carry: dict | None = None
        if state:
            sampler.position = state["position"]
            progress.report(active_before_s=self.active_before)

    # -- state of the other callbacks -------------------------------------------------------
    def _callback_states(self) -> dict:
        states = {}
        for name, cb in self.tracked.items():
            if hasattr(cb, "get_state"):
                states[name] = cb.get_state()
            elif isinstance(cb, keras.callbacks.EarlyStopping):
                best = None if cb.best is None else float(cb.best)
                states[name] = {"wait": cb.wait, "best": best, "best_epoch": cb.best_epoch,
                                "stopped_epoch": cb.stopped_epoch}  # fmt: skip
            elif hasattr(cb, "_step"):  # MLflowMetricsLogger
                states[name] = {"step": cb._step}
        return states

    def _set_callback_states(self, states: dict, from_epoch: int | None = None) -> None:
        from rsseg.train.callbacks import TopKCheckpoint

        for name, cb in self.tracked.items():
            if name not in states:
                continue
            s = states[name]
            if isinstance(cb, TopKCheckpoint):
                cb.set_state(s, from_epoch)
            elif hasattr(cb, "set_state"):
                cb.set_state(s)
            elif isinstance(cb, keras.callbacks.EarlyStopping):
                cb.wait, cb.best_epoch, cb.stopped_epoch = (
                    s["wait"],
                    s["best_epoch"],
                    s["stopped_epoch"],
                )
                cb.best = s["best"]
            elif hasattr(cb, "_step"):
                cb._step = s["step"]

    # -- Keras hooks ------------------------------------------------------------------------
    def on_train_begin(self, logs=None):
        opt = self.model.optimizer
        if not opt.built:
            opt.build(self.model.trainable_variables)
        if self._restore_all:
            arrays = self.state["arrays"]
            _assign(self.model.variables, arrays["model"], "model")
            _assign(opt.variables, arrays["optimizer"], "optimizer")
            self._set_callback_states(self.state["callbacks"], from_epoch=self.state["epoch"])
            _truncate_csv(self.run_dir / "history.csv", self.state["epoch"])
            self._restore_all = False
            log.info("resumed from epoch %d, batch %d", self.epoch + 1, self.batch)
        elif self._carry is not None:  # second fit() of a run resumed mid-epoch
            self._set_callback_states(self._carry)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if self._restore_metrics:  # after fit() reset the metrics of the resumed epoch
            _assign(self.model.metrics_variables, self.state["arrays"]["metrics"], "metrics")
            self._restore_metrics = False
        else:
            self.batch = 0

    def on_train_batch_end(self, batch, logs=None):
        self.sampler.position += self.sampler.batch_size
        self.batch += 1
        action = control.pending()
        if action == "stop":
            raise control.StopRequested(run_dir=self.run_dir)
        if action == "pause":
            self.save("pause", mid_epoch=True)
            raise control.PauseRequested(run_dir=self.run_dir)
        if self.every_s and time.time() - self.last_save >= self.every_s:
            self.save("auto", mid_epoch=True)

    def on_test_batch_end(self, batch, logs=None):
        if control.pending() == "stop":  # a pause waits for the end of the epoch
            raise control.StopRequested(run_dir=self.run_dir)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch, self.batch = epoch + 1, 0
        action = control.pending()
        if action == "stop":
            raise control.StopRequested(run_dir=self.run_dir)
        if action == "pause":
            self.save("pause", mid_epoch=False)
            raise control.PauseRequested(run_dir=self.run_dir)
        if self.every_s and time.time() - self.last_save >= self.every_s:
            self.save("auto", mid_epoch=False)

    def on_train_end(self, logs=None):
        self._carry = self._callback_states()

    # -- checkpoint -------------------------------------------------------------------------
    def active_seconds(self) -> float:
        return self.active_before + time.time() - self.t_start

    def save(self, reason: str, mid_epoch: bool) -> Path:
        """Write ``state.npz`` then ``state.json`` (each atomically, matched by a generation)."""
        d = resume_dir(self.run_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.generation += 1
        groups = {
            "model": _arrays(self.model.variables),
            "optimizer": _arrays(self.model.optimizer.variables),
            "metrics": _arrays(self.model.metrics_variables) if mid_epoch else [],
        }
        payload: dict[str, np.ndarray] = {"generation": np.array(self.generation)}
        for kind, arrays in groups.items():
            payload[f"n_{kind}"] = np.array(len(arrays))
            payload |= {f"{kind}{i}": a for i, a in enumerate(arrays)}
        tmp = d / "state.tmp.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, d / STATE_ARRAYS)
        meta = {
            "kind": self.kind,
            "phase": "train",
            "epoch": self.epoch,
            "batch": self.batch if mid_epoch else 0,
            "epochs": self.epochs,
            "steps": self.steps,
            "position": self.sampler.position,
            "early_stopped": bool(self.model.stop_training),
            "run_id": self.run_id,
            "reason": reason,
            "saved": time.time(),
            "active_s": self.active_seconds(),
            "generation": self.generation,
            "callbacks": self._callback_states(),
            "env": {**environment(), "gpu": gpu_name()},
        }
        _write_json(d / STATE_JSON, meta)
        self.last_save = time.time()
        progress.report(checkpoint=meta["saved"], checkpoint_reason=reason)
        log.info("resume checkpoint (%s) at epoch %d, batch %d", reason, self.epoch + 1,
                 meta["batch"])  # fmt: skip
        return d


def fit_resumable(model: keras.Model, loader, epochs: int, steps: int,
                  callbacks: list, resume: ResumeCheckpoint, reporter=None, **fit_kwargs) -> None:  # fmt: skip
    """``model.fit`` for ``epochs`` of ``steps`` batches, from the position of ``resume``.

    A run resumed mid-epoch first completes that epoch (``steps - batch`` batches), then a
    second ``fit`` runs the remaining epochs. The training stream must come from a
    :class:`rsseg.data.datasets.StreamSampler`, so both calls see the batches of an
    uninterrupted run.
    """
    start, done = resume.epoch, resume.batch
    if resume.state and resume.state.get("early_stopped"):
        return
    if done:
        if reporter is not None:
            reporter.batch_offset = done
        model.fit(loader, initial_epoch=start, epochs=start + 1, steps_per_epoch=steps - done,
                  callbacks=callbacks, **fit_kwargs)  # fmt: skip
        start += 1
        if model.stop_training:
            return
    if start < epochs:
        model.fit(loader, initial_epoch=start, epochs=epochs, steps_per_epoch=steps,
                  callbacks=callbacks, **fit_kwargs)  # fmt: skip
