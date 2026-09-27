# SPDX-License-Identifier: AGPL-3.0-or-later
"""From the progress of a job to what the application shows: phases, progress bars, elapsed
and remaining time, expected end.

The remaining time combines two sources:

* **before and at the start** of a job, the estimate of :class:`rsseg.eta.Estimator` (times
  measured on an RTX 4060, recalibrated on the finished campaign steps);
* **while it runs**, the measured duration of the epochs, tiles or images: the estimate
  corrects itself as the job goes on. With early stopping the remaining time is a range, from
  "stops as early as possible" to "runs every epoch".

The elapsed time counts only work: pauses are excluded, the time before a resume is included.
This module does not import Qt, torch or keras.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from rsseg.gui.core.jobs import STATE_LABELS, Job, elapsed_seconds

STALE_S = 15 * 60
"""A running job whose progress has not changed for this long is flagged."""

PHASES: dict[str, list[tuple[str, str | None]]] = {
    "train": [("Avvio", "startup"), ("Training", "train"), ("Valutazione test", "evaluation"),
              ("Report", None)],
    "pretrain": [("Avvio", "startup"), ("Pre-training DINO", "pretrain"), ("Esportazione", None)],
    "evaluate": [("Avvio", "startup"), ("Valutazione", "evaluation"), ("Report", None)],
    "predict": [("Caricamento modello", "startup"), ("Inferenza", "predict"), ("Salvataggio", None)],
    "prepare": [("Avvio", "startup"), ("Preparazione", "prepare"), ("Manifest", None)],
    "prepare-unlabeled": [("Avvio", "startup"), ("Ritaglio dei frame", "prepare"),
                          ("Manifest", None)],
    "campaign": [("Pianificazione", "startup"), ("Dati", "prepare"), ("Esperimenti", "campaign"),
                 ("Report", None)],
}  # fmt: skip
DEFAULT_PHASES = [("Avvio", "startup"), ("Esecuzione", "run"), ("Fine", None)]


@dataclass
class Bar:
    label: str
    fraction: float | None
    """``None``: indeterminate (the total is not known yet)."""
    text: str = ""


@dataclass
class JobView:
    title: str
    state: str
    state_label: str
    phases: list[tuple[str, str]] = field(default_factory=list)
    """``(label, status)`` with status ``done``, ``active`` or ``todo``."""
    bars: list[Bar] = field(default_factory=list)
    elapsed_s: float = 0.0
    remaining_s: float | None = None
    remaining_max_s: float | None = None
    """Upper end of the remaining time when it is a range (early stopping)."""
    finish_at: float | None = None
    notes: list[str] = field(default_factory=list)
    stale: bool = False
    history: list[dict] = field(default_factory=list)

    @property
    def total_fraction(self) -> float | None:
        if self.state == "done":
            return 1.0
        if self.remaining_s is None:
            return None
        total = self.elapsed_s + self.remaining_s
        return self.elapsed_s / total if total > 0 else None


# ------------------------------------------------------------------------------ formatting
def fmt_duration(seconds: float | None) -> str:
    """``1 h 05 min``, ``4 min 10 s``, ``12 s`` (``—`` if unknown)."""
    if seconds is None or not math.isfinite(seconds):
        return "—"
    s = max(int(round(seconds)), 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d} g {h} h"
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {sec:02d} s"
    return f"{sec} s"


def fmt_clock(ts: float | None, now: float | None = None) -> str:
    """Time of day of ``ts`` (with the date if it is not today)."""
    if ts is None or not math.isfinite(ts):
        return "—"
    t, n = time.localtime(ts), time.localtime(now or time.time())
    if (t.tm_year, t.tm_yday) == (n.tm_year, n.tm_yday):
        return time.strftime("%H:%M", t)
    return time.strftime("%d/%m %H:%M", t)


def fmt_range(low: float | None, high: float | None) -> str:
    if low is None:
        return "—"
    if high is None or abs(high - low) < 60:
        return f"≈ {fmt_duration(low)}"
    return f"{fmt_duration(low)} – {fmt_duration(high)}"


def _finite(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _frac(a, b) -> float | None:
    if not b:
        return None
    return min(max(float(a or 0) / float(b), 0.0), 1.0)


# ------------------------------------------------------------------------------- phases
def _phases(command: str, p: dict, state: str) -> list[tuple[str, str]]:
    spec = PHASES.get(command, DEFAULT_PHASES)
    keys = [k for _, k in spec]
    phase = p.get("phase") or "startup"
    if command in ("train", "evaluate") and phase == "train":
        idx = 1
    elif phase in keys:
        idx = keys.index(phase)
    elif phase == "evaluation" and "evaluation" in keys:
        idx = keys.index("evaluation")
    else:
        idx = 1 if phase != "startup" else 0
    if state == "done":
        idx = len(spec)
    out = []
    for i, (label, _) in enumerate(spec):
        status = "done" if i < idx else ("active" if i == idx and state not in
                                         ("failed", "interrupted", "stopped") else "todo")  # fmt: skip
        if i == idx and state in ("failed", "interrupted", "stopped"):
            status = "error"
        out.append((label, status))
    return out


# ------------------------------------------------------------------------------ describe
def describe(job: Job, p: dict | None, now: float | None = None, cfg=None, estimator=None,
             step_progress: dict | None = None, step_cfg=None) -> JobView:  # fmt: skip
    """What to show for ``job`` given its progress ``p``.

    ``cfg`` (the configuration of the job) and ``estimator`` refine the estimates before the
    measurements exist; ``step_progress`` is the progress of the running step of a campaign.
    """
    now = now or time.time()
    p = p or {}
    view = JobView(job.title, job.state, STATE_LABELS.get(job.state, job.state))
    view.elapsed_s = elapsed_seconds(job, p, now)
    view.phases = _phases(job.command, p, job.state)
    view.history = list(p.get("history") or [])
    if job.state == "queued":
        view.remaining_s = job.estimate_s
        view.notes.append("in attesa che la GPU si liberi" if job.uses_gpu else "in avvio")
        return view
    running = job.state in ("running", "pausing", "stopping")

    if job.command in ("train", "pretrain", "evaluate"):
        low, high = _train_bars(view, job, p, now, cfg, estimator)
    elif job.command == "campaign":
        low, high = _campaign_bars(view, p, now, step_progress, step_cfg, estimator)
    elif job.command == "predict":
        low, high = _predict_bars(view, p, now)
    else:
        low, high = _count_bars(view, p, now)

    if low is None and job.estimate_s and p.get("phase") in (None, "startup"):
        low = max(job.estimate_s - view.elapsed_s, 0.0)
    if running:
        view.remaining_s = low
        view.remaining_max_s = high if high is not None and low is not None and high > low else None
        if low is not None:
            view.finish_at = now + low
        updated = p.get("updated")
        if updated and now - updated > STALE_S:
            view.stale = True
            view.notes.append(f"nessun avanzamento da {fmt_duration(now - updated)}: "
                              "controlla il log")  # fmt: skip
    elif job.state == "done":
        view.remaining_s = 0.0
    if job.state == "pausing":
        view.notes.append("pausa richiesta: il job salva un checkpoint e si ferma "
                          "(durante la validazione, alla fine dell'epoca)")  # fmt: skip
    if job.state == "stopping":
        view.notes.append("arresto richiesto")
    if job.state == "paused":
        since = job.finished or p.get("finished")
        view.notes.append(f"in pausa da {fmt_duration(now - since) if since else '—'}")
    if p.get("checkpoint"):
        view.notes.append(f"ultimo checkpoint di ripresa: {fmt_clock(p['checkpoint'], now)}")
    if job.error and job.state in ("failed", "interrupted"):
        view.notes.append(job.error)
    return view


def _train_bars(view: JobView, job: Job, p: dict, now: float, cfg, estimator):
    epoch_prior = eval_prior = None
    if cfg is not None and estimator is not None:
        if job.command == "pretrain":
            epoch_prior = _finite(estimator.dino_epoch_s(cfg))
            eval_prior = 0.0
        else:
            cost = estimator.train_cost(cfg)
            epoch_prior, eval_prior = _finite(cost.epoch_s), _finite(cost.eval_s)
    if job.command == "evaluate":
        eval_prior = _finite(estimator.eval_seconds(cfg)) if cfg and estimator else None

    phase = p.get("phase")
    if phase == "evaluation":
        tile, tiles = p.get("tile") or 1, p.get("tiles") or 0
        view.bars.append(
            Bar("Tile", _frac(tile - 1, tiles), f"{tile}/{tiles} · {p.get('stage', '')}")
        )
        tile_s = _finite(p.get("tile_s")) or (eval_prior / tiles if eval_prior and tiles else None)
        if tile_s is None:
            return None, None
        in_tile = max(now - (p.get("updated") or now), 0.0)
        return max((tiles - tile + 1) * tile_s - min(in_tile, tile_s), 0.0), None
    if phase not in ("train", "pretrain"):
        return None, None

    epoch, epochs = p.get("epoch") or 0, p.get("epochs") or 0
    batch, steps = p.get("batch") or 0, p.get("steps") or 0
    validating = p.get("stage") == "validation"
    view.bars.append(Bar("Epoca", _frac(max(epoch - 1, 0), epochs), f"{epoch}/{epochs}"))
    batch_frac = 1.0 if validating else _frac(batch, steps)
    view.bars.append(Bar("Batch", batch_frac,
                         "validazione…" if validating else f"{batch}/{steps}"))  # fmt: skip
    if p.get("best_epoch") and p.get("monitor"):
        note = f"{p['monitor']} migliore {p['best']:.4f} (epoca {p['best_epoch']})"
        if p.get("patience"):
            since = p.get("since_best", 0)
            note += f" · early stopping: {since}/{p['patience']} epoche senza miglioramenti"
        view.notes.append(note)
    elif p.get("loss") is not None:
        view.notes.append(f"loss {p['loss']:.4f}")

    epoch_s = _finite(p.get("epoch_s")) or epoch_prior
    if epoch_s is None or not epochs:
        return None, None
    done = max(epoch - 1, 0) + (batch_frac or 0.0)
    eval_s = eval_prior or 0.0
    high = max(epochs - done, 0.0) * epoch_s + eval_s
    low = high
    if p.get("patience") and p.get("best_epoch"):
        earliest = min(epochs, max(p["best_epoch"] + p["patience"], epoch))
        low = max(earliest - done, 0.0) * epoch_s + eval_s
    return low, high


def _count_bars(view: JobView, p: dict, now: float):
    done, total = p.get("done"), p.get("total")
    if total:
        label = {"tiles": "Tile", "frames": "Frame", "inspection": "Tile"}.get(
            p.get("stage", ""), "Avanzamento"
        )
        view.bars.append(Bar(label, _frac(done, total), f"{done or 0}/{total}"))
        if done:
            spent = now - p.get("started", now)
            return spent / done * (total - done), None
    elif p.get("phase") not in (None, "startup"):
        view.bars.append(Bar("Avanzamento", None, p.get("stage", "")))
    return None, None


def _predict_bars(view: JobView, p: dict, now: float):
    done, total = p.get("done") or 0, p.get("total") or 0
    window, windows = p.get("window") or 0, p.get("windows") or 0
    if not total:
        return None, None
    view.bars.append(Bar("Immagini", _frac(done, total), f"{done}/{total} · {p.get('stage', '')}"))
    view.bars.append(Bar("Finestre", _frac(window, windows), f"{window}/{windows or '?'}"))
    work = done + (_frac(window, windows) or 0.0)
    if work <= 0:
        return None, None
    spent = now - p.get("started", now)
    return spent / work * (total - work), None


def _campaign_bars(view: JobView, p: dict, now: float, step_p: dict | None, step_cfg, estimator):
    if p.get("phase") == "prepare":
        return _count_bars(view, p, now)
    done, total = p.get("done") or 0, p.get("total") or 0
    if not total:
        return None, None
    view.bars.append(Bar("Passi", _frac(done, total), f"{done}/{total} · {p.get('stage', '')}"))
    remaining = _finite(p.get("remaining_estimate_s"))
    low = high = None
    if remaining is not None:
        low = max(remaining - (now - (p.get("step_started") or now)), 0.0)
    if step_p:
        kind = p.get("step_kind", "train")
        sub = Job("step", "pretrain" if kind == "pretrain" else "train", [], p.get("stage", ""),
                  state="running")  # fmt: skip
        sub_view = describe(sub, step_p, now, step_cfg, estimator)
        for b in sub_view.bars:
            view.bars.append(Bar(f"  {b.label}", b.fraction, b.text))
        view.notes += sub_view.notes
        step_est = _finite(p.get("step_estimate_s"))
        if sub_view.remaining_s is not None and remaining is not None and step_est is not None:
            low = max(remaining - step_est, 0.0) + sub_view.remaining_s
            if sub_view.remaining_max_s is not None:
                high = max(remaining - step_est, 0.0) + sub_view.remaining_max_s
    return low, high
