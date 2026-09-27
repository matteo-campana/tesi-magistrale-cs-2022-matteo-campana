# SPDX-License-Identifier: AGPL-3.0-or-later
"""State shared by the pages: project folder, job queue, GPU monitor, estimates."""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QThread, QTimer, Signal, Slot

from rsseg.config import ExperimentConfig, load_config
from rsseg.eta import Estimator
from rsseg.gui.core.gpu import TORCH_CHECK, GPULog, GPUProbe, GPUSample
from rsseg.gui.core.jobs import Job, JobStore
from rsseg.gui.core.timing import JobView, describe

HISTORY_S = 600
"""Seconds of GPU samples kept for the charts."""


class GPUWorker(QObject):
    """Samples the GPU once a second in its own thread (``nvidia-smi`` can take 50 ms)."""

    sampled = Signal(object)
    ready = Signal(str, object, object)  # mode, driver, cuda driver

    def __init__(self, interval_ms: int = 1000):
        super().__init__()
        self.interval_ms = interval_ms
        self.probe: GPUProbe | None = None
        self.timer: QTimer | None = None

    @Slot()
    def start(self) -> None:
        self.probe = GPUProbe()
        self.ready.emit(self.probe.mode, self.probe.driver, self.probe.cuda_driver)
        self.timer = QTimer()
        self.timer.timeout.connect(self.sample)
        self.timer.start(self.interval_ms)
        self.sample()

    @Slot()
    def sample(self) -> None:
        if self.probe is not None:
            self.sampled.emit(self.probe.sample())

    @Slot()
    def stop(self) -> None:
        if self.timer is not None:
            self.timer.stop()


class AppContext(QObject):
    jobs_changed = Signal()
    """Every second (times move) and whenever a job changes state."""
    job_finished = Signal(object)
    gpu_sampled = Signal(object)
    gpu_ready = Signal()
    torch_checked = Signal(dict)
    status_message = Signal(str)

    def __init__(self, root: str | Path, gpu: bool = True, tick_ms: int = 1000):
        super().__init__()
        self.root = Path(root).resolve()
        self.store = JobStore(self.root)
        self.samples: list[GPUSample] = []
        self.gpu_history: deque[GPUSample] = deque(maxlen=HISTORY_S)
        self.gpu_mode = "none"
        self.driver: str | None = None
        self.cuda_driver: str | None = None
        self.torch_info: dict | None = None
        self._configs: dict[str, ExperimentConfig | None] = {}
        self._estimator: Estimator | None = None
        self._estimator_time = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(tick_ms)

        self.gpu_thread: QThread | None = None
        if gpu:
            self.gpu_thread = QThread(self)
            self.gpu_worker = GPUWorker()
            self.gpu_worker.moveToThread(self.gpu_thread)
            self.gpu_thread.started.connect(self.gpu_worker.start)
            self.gpu_worker.sampled.connect(self._on_samples)
            self.gpu_worker.ready.connect(self._on_gpu_ready)
            self.gpu_thread.start()
            self.check_torch()

    # ------------------------------------------------------------------------ GPU
    def check_torch(self) -> None:
        """Ask a separate Python process what PyTorch sees (asynchronous)."""
        self.torch_info = None
        proc = QProcess(self)
        proc.setWorkingDirectory(str(self.root))

        def done(*_):
            out = bytes(proc.readAllStandardOutput()).decode(errors="replace").strip()
            err = bytes(proc.readAllStandardError()).decode(errors="replace").strip()
            try:
                self.torch_info = json.loads(out.splitlines()[-1])
            except (ValueError, IndexError):
                self.torch_info = {"error": (err.splitlines() or ["torch non importabile"])[-1]}
            self.torch_checked.emit(self.torch_info)
            proc.deleteLater()

        proc.finished.connect(done)
        proc.start(self.store.python, ["-c", TORCH_CHECK])

    @Slot(str, object, object)
    def _on_gpu_ready(self, mode, driver, cuda):
        self.gpu_mode, self.driver, self.cuda_driver = mode, driver, cuda
        self.gpu_ready.emit()

    @Slot(object)
    def _on_samples(self, samples: list[GPUSample]) -> None:
        self.samples = samples
        if samples:
            self.gpu_history.append(samples[0])
            for job in self.store.jobs.values():
                if job.uses_gpu and job.state in ("running", "pausing", "stopping"):
                    GPULog(self.store.folder(job) / "gpu_log.csv").append(samples[0])
        self.gpu_sampled.emit(samples)

    @property
    def sample(self) -> GPUSample | None:
        return self.samples[0] if self.samples else None

    def own_pids(self) -> set[int]:
        pids = {j.pid for j in self.store.jobs.values() if j.pid and j.active}
        try:  # workers of the DataLoader are children of the job
            import psutil

            for pid in list(pids):
                with contextlib.suppress(psutil.Error):
                    pids |= {c.pid for c in psutil.Process(pid).children(recursive=True)}
        except ImportError:
            pass
        return pids

    def shutdown(self) -> None:
        self.timer.stop()
        if self.gpu_thread is not None:
            self.gpu_worker.stop()
            self.gpu_thread.quit()
            self.gpu_thread.wait(3000)

    # ----------------------------------------------------------------------- jobs
    @Slot()
    def tick(self) -> None:
        before = {j.id: j.state for j in self.store.jobs.values()}
        self.store.tick()
        for job in self.store.jobs.values():
            if before.get(job.id) in ("running", "pausing", "stopping") and not job.active:
                self._finished(job)
        self.jobs_changed.emit()

    def _finished(self, job: Job) -> None:
        log = self.store.folder(job) / "gpu_log.csv"
        if job.run_dir and log.exists() and Path(job.run_dir).exists():
            with contextlib.suppress(OSError):
                shutil.copyfile(log, Path(job.run_dir) / "gpu_log.csv")
        if job.command == "benchmark" and job.state == "done":
            result = self.store.result(job)
            if isinstance(result, dict) and job.meta.get("bench_key"):
                self.save_benchmark(job.meta["bench_key"], result)
        self._estimator = None  # measurements may have changed
        self.job_finished.emit(job)

    def submit(self, command: str, args: list[str], title: str, config: str | None = None,
               overrides: list[str] | None = None, meta: dict | None = None) -> Job:  # fmt: skip
        cfg = self.config_for(config, overrides or []) if config else None
        estimate = self.estimate(command, cfg) if cfg is not None else None
        job = self.store.add(command, args, title, config=config, overrides=list(overrides or []),
                             estimate_s=estimate, meta=meta or {})  # fmt: skip
        self.tick()
        self.status_message.emit(f"«{title}» aggiunto alla coda")
        return job

    def view(self, job: Job, now: float | None = None) -> JobView:
        p = self.store.progress(job)
        cfg = self.job_config(job)
        step_p = step_cfg = None
        if job.command == "campaign" and p and p.get("step_progress"):
            from rsseg.progress import read

            step_p = read(p["step_progress"])
        return describe(job, p, now or time.time(), cfg, self.estimator(), step_p, step_cfg)

    # ------------------------------------------------------------ configs, estimates
    def config_for(self, path: str | None, overrides: list[str]) -> ExperimentConfig | None:
        key = json.dumps([path, overrides])
        if key not in self._configs:
            try:
                p = Path(path) if path else None
                if p is not None and not p.is_absolute():
                    p = self.root / p
                cfg = load_config(p, overrides)
                # relative data paths are relative to the project (the cwd of the jobs)
                for attr in ("root", "raw_dir"):
                    value = getattr(cfg.data, attr)
                    if value and not Path(value).is_absolute():
                        setattr(cfg.data, attr, str(self.root / value))
                self._configs[key] = cfg
            except Exception:  # noqa: BLE001 - shown as "no estimate"
                self._configs[key] = None
        return self._configs[key]

    def job_config(self, job: Job) -> ExperimentConfig | None:
        if job.command in ("train", "pretrain") and job.run_dir and "--resume" in job.args:
            return self.config_for(str(Path(job.run_dir) / "config.yaml"), [])
        if job.config:
            return self.config_for(job.config, job.overrides)
        return None

    def estimator(self) -> Estimator:
        if self._estimator is None or time.time() - self._estimator_time > 300:
            store = self.root / "runs" / "campaigns" / "store"
            self._estimator = Estimator(store if store.exists() else None)
            self._estimator_time = time.time()
        return self._estimator

    def estimate(self, command: str, cfg: ExperimentConfig | None) -> float | None:
        if cfg is None or command not in ("train", "pretrain", "benchmark"):
            return None
        s = self.estimator().seconds(command, cfg)
        return s if s == s else None  # nan → None

    # ---------------------------------------------------------------- benchmarks
    @property
    def benchmarks_path(self) -> Path:
        return self.store.dir / "benchmarks.json"

    def benchmarks(self) -> dict:
        try:
            return json.loads(self.benchmarks_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_benchmark(self, key: str, result: dict) -> None:
        data = self.benchmarks()
        data[key] = {**result, "measured": time.time()}
        self.benchmarks_path.parent.mkdir(parents=True, exist_ok=True)
        self.benchmarks_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def bench_key(config: str | None, overrides: list[str], mode: str = "seg") -> str:
        return json.dumps([mode, config, sorted(overrides)])

    def benchmark_peak_gb(self, config: str | None, overrides: list[str],
                          mode: str = "seg") -> float | None:  # fmt: skip
        r = self.benchmarks().get(self.bench_key(config, overrides, mode))
        if not r:
            return None
        for k in ("peak_gpu_memory_gb", "gpu_peak_mem_gb"):
            if r.get(k) is not None:
                return float(r[k])
        return None
