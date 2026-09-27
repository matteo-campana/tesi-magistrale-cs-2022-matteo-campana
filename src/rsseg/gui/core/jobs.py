# SPDX-License-Identifier: AGPL-3.0-or-later
"""Jobs of the application: ``rsseg`` commands run as detached processes.

Every job has a folder ``<root>/.gui/jobs/<id>/`` with ``job.json`` (the record below),
``progress.json`` and ``control.json`` (:mod:`rsseg.progress`, :mod:`rsseg.control`),
``log.txt`` (stderr), ``stdout.json`` (the ``--json`` result) and ``gpu_log.csv``.

Processes are started detached (their own session / process group), so they outlive the
application: closing it can leave a training running in the background, and the next start
finds it again (:meth:`JobStore.refresh`). A job that disappears without writing its final
state (crash, power loss, killed) becomes ``interrupted``; if its run has a resume checkpoint
it can be resumed like a paused one.

GPU jobs (training, pre-training, evaluation, prediction, benchmark, campaign) run one at a
time — two trainings do not fit in 8 GB — the others (data preparation, synthetic data) start
at once. This module does not import Qt, torch or keras.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from rsseg import control, progress
from rsseg.runstate import is_resumable

GPU_COMMANDS = {"train", "pretrain", "evaluate", "predict", "benchmark", "campaign"}
PAUSABLE = {"train", "pretrain", "campaign"}
ACTIVE = ("queued", "running", "pausing", "stopping")
FINAL = ("done", "failed", "paused", "stopped", "interrupted")

STATE_LABELS = {
    "queued": "in coda",
    "running": "in esecuzione",
    "pausing": "pausa richiesta…",
    "stopping": "arresto richiesto…",
    "done": "completato",
    "failed": "fallito",
    "paused": "in pausa",
    "stopped": "interrotto",
    "interrupted": "terminato in modo anomalo",
}


@dataclass
class Job:
    id: str
    command: str
    """Sub-command of ``rsseg`` (``train``, ``prepare``, …)."""
    args: list[str]
    """Arguments after the sub-command."""
    title: str
    state: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    pid: int | None = None
    exit_code: int | None = None
    run_dir: str | None = None
    """Run folder written by the command (from its progress), used to resume it."""
    config: str | None = None
    overrides: list[str] = field(default_factory=list)
    estimate_s: float | None = None
    """Duration expected before the start (:class:`rsseg.eta.Estimator`)."""
    sessions: int = 0
    """How many times the process was started (1 + resumes)."""
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def uses_gpu(self) -> bool:
        return self.command in GPU_COMMANDS and not (
            self.command in ("campaign",) and "--dry-run" in self.args
        )

    @property
    def pausable(self) -> bool:
        return self.command in PAUSABLE

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    @property
    def resumable(self) -> bool:
        if self.state not in ("paused", "interrupted", "stopped"):
            return False
        if self.command == "campaign":
            return self.state in ("paused", "interrupted")
        return self.state != "stopped" and bool(self.run_dir) and is_resumable(self.run_dir)


def new_job_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def pid_alive(pid: int | None) -> bool:
    """Whether process ``pid`` exists and has not exited (zombies count as exited)."""
    if not pid:
        return False
    try:
        import psutil

        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - psutil.NoSuchProcess, AccessDenied
        return False
    if os.name == "nt":  # pragma: no cover - Windows without psutil
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie child of this process: reap it
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


class JobStore:
    """The jobs of a project folder, persisted in ``<root>/.gui/jobs.json``."""

    def __init__(self, root: str | Path, python: str | None = None):
        self.root = Path(root)
        self.dir = self.root / ".gui"
        self.python = python or sys.executable
        self.jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self.load()

    # ---------------------------------------------------------------- persistence
    @property
    def path(self) -> Path:
        return self.dir / "jobs.json"

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        names = {f for f in Job.__dataclass_fields__}
        self.jobs = {d["id"]: Job(**{k: v for k, v in d.items() if k in names}) for d in data}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = [asdict(j) for j in sorted(self.jobs.values(), key=lambda j: j.created)]
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def folder(self, job: Job) -> Path:
        return self.dir / "jobs" / job.id

    def progress_path(self, job: Job) -> Path:
        return self.folder(job) / progress.FILE_NAME

    def control_path(self, job: Job) -> Path:
        return self.folder(job) / control.FILE_NAME

    def log_path(self, job: Job) -> Path:
        return self.folder(job) / "log.txt"

    def result_path(self, job: Job) -> Path:
        return self.folder(job) / "stdout.json"

    def progress(self, job: Job) -> dict | None:
        return progress.read(self.progress_path(job))

    def result(self, job: Job) -> dict | list | None:
        """The JSON printed by the command (``--json``), if it finished."""
        try:
            text = self.result_path(job).read_text(encoding="utf-8")
        except OSError:
            return None
        for start in (text.find("{"), text.find("[")):
            if start >= 0:
                try:
                    return json.loads(text[start:])
                except ValueError:
                    continue
        return None

    def log_tail(self, job: Job, lines: int = 200) -> str:
        try:
            text = self.log_path(job).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    # -------------------------------------------------------------------- queue
    def add(self, command: str, args: list[str], title: str, **kw) -> Job:
        job = Job(new_job_id(), command, list(args), title, **kw)
        self.jobs[job.id] = job
        self.save()
        return job

    def remove(self, job: Job) -> None:
        if job.active and job.state != "queued":
            raise ValueError("stop the job before removing it")
        self.jobs.pop(job.id, None)
        self.save()

    def ordered(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def gpu_busy(self) -> Job | None:
        return next((j for j in self.jobs.values()
                     if j.uses_gpu and j.state in ("running", "pausing", "stopping")), None)  # fmt: skip

    def command_line(self, job: Job) -> list[str]:
        return [self.python, "-m", "rsseg.cli", "--json", job.command, *job.args]

    def start(self, job: Job) -> None:
        """Start ``job`` now as a detached process (the queue calls it; see :meth:`tick`)."""
        folder = self.folder(job)
        folder.mkdir(parents=True, exist_ok=True)
        self.progress_path(job).unlink(missing_ok=True)
        control.clear(self.control_path(job))
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               progress.ENV: str(self.progress_path(job)),
               control.ENV: str(self.control_path(job))}  # fmt: skip
        env.pop("QT_QPA_PLATFORM", None)
        kwargs: dict = {}
        if os.name == "nt":  # pragma: no cover
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008
        else:
            kwargs["start_new_session"] = True
        with self.log_path(job).open("a", encoding="utf-8") as err, \
                self.result_path(job).open("w", encoding="utf-8") as out:  # fmt: skip
            err.write(f"\n$ {subprocess.list2cmdline(self.command_line(job))}\n")
            err.flush()
            proc = subprocess.Popen(self.command_line(job), cwd=self.root, env=env,
                                    stdout=out, stderr=err, stdin=subprocess.DEVNULL, **kwargs)  # fmt: skip
        self._procs[job.id] = proc
        job.pid, job.state, job.started, job.finished = proc.pid, "running", time.time(), None
        job.exit_code, job.error = None, None
        job.sessions += 1
        self.save()

    def request(self, job: Job, action: str) -> None:
        """Ask a running job to ``pause`` or ``stop``; a queued one is simply dropped."""
        if job.state == "queued":
            job.state = "stopped"
            self.save()
            return
        if job.state not in ("running", "pausing", "stopping"):
            return
        if action == "pause" and not job.pausable:
            action = "stop"
        control.request(self.control_path(job), action)
        job.state = "pausing" if action == "pause" else "stopping"
        self.save()

    def kill(self, job: Job) -> None:
        """Terminate a job that does not answer a stop request."""
        if job.pid and pid_alive(job.pid):
            try:
                if os.name == "nt":  # pragma: no cover
                    os.kill(job.pid, signal.SIGTERM)
                else:
                    os.killpg(os.getpgid(job.pid), signal.SIGTERM)
            except OSError:
                pass

    def resume(self, job: Job) -> Job:
        """Queue a paused or interrupted job again: training continues from its checkpoint."""
        if not job.resumable:
            raise ValueError(f"job {job.title!r} cannot be resumed")
        if job.command in ("train", "pretrain"):
            job.args = ["--resume", str(job.run_dir)]
        # campaigns resume by running again: finished steps are skipped, a paused one resumes
        job.state = "queued"
        self.save()
        return job

    def tick(self) -> list[Job]:
        """Update the state of every job and start the queued ones; returns the changed jobs."""
        changed = []
        for job in self.jobs.values():
            if job.state in ("running", "pausing", "stopping") and self._update(job):
                changed.append(job)
        for job in sorted(self.jobs.values(), key=lambda j: j.created):
            if job.state != "queued":
                continue
            if job.uses_gpu and self.gpu_busy() is not None:
                continue
            self.start(job)
            changed.append(job)
        if changed:
            self.save()
        return changed

    def refresh(self) -> list[Job]:
        """After a restart: reattach to jobs still running, settle the others."""
        return self.tick()

    def _update(self, job: Job) -> bool:
        p = self.progress(job) or {}
        if p.get("run_dir"):
            run_dir = str((self.root / p["run_dir"]).resolve())  # relative to the job's cwd
            if run_dir != job.run_dir:
                job.run_dir = run_dir
        proc = self._procs.get(job.id)
        if proc is not None:
            code = proc.poll()
            alive = code is None
        else:
            code, alive = None, pid_alive(job.pid) and _same_process(job)
        if alive:
            return False
        self._procs.pop(job.id, None)
        job.finished = p.get("finished") or time.time()
        job.exit_code = code if code is not None else p.get("exit_code")
        state = p.get("state")
        if code == control.EXIT_PAUSED:
            state = "paused"
        elif code == control.EXIT_STOPPED:
            state = "stopped"
        elif code is not None and code != 0 and state not in ("failed",):
            state = "failed"
        if state not in FINAL or state == "running" or state is None:
            state = "interrupted"
        job.state = state
        job.error = p.get("error")
        if state == "failed" and not job.error:
            last = self.log_tail(job, 5).strip().splitlines()
            job.error = last[-1] if last else None
        return True


def _same_process(job: Job) -> bool:
    """Guard against a reused PID: the process must not be younger than the job."""
    try:
        import psutil

        return psutil.Process(job.pid).create_time() <= (job.started or 0) + 5
    except Exception:  # noqa: BLE001 - no psutil, or the process vanished
        return True


def elapsed_seconds(job: Job, p: dict | None, now: float | None = None) -> float:
    """Active seconds of work of ``job`` (pauses excluded), including earlier sessions."""
    now = now or time.time()
    if p and p.get("started"):
        end = p.get("finished") if p.get("state") in progress.FINAL_STATES else None
        return p.get("active_before_s", 0.0) + (end or now) - p["started"]
    if job.started:
        return (job.finished or now) - job.started
    return 0.0
