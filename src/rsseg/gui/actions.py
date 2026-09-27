# SPDX-License-Identifier: AGPL-3.0-or-later
"""User actions shared by several pages (with their confirmation dialogs)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox, QWidget

from rsseg.gui.context import AppContext
from rsseg.gui.core.gpu import preflight
from rsseg.gui.core.jobs import Job
from rsseg.gui.core.timing import fmt_duration
from rsseg.runstate import environment_changes, read_state


def confirm_gpu_start(parent: QWidget, ctx: AppContext, needed_gb: float | None,
                      what: str) -> bool:  # fmt: skip
    """Warn before a GPU job (no GPU, busy memory, benchmark missing or too big)."""
    warnings = preflight(ctx.sample, needed_gb, ctx.own_pids(), ctx.store.gpu_busy() is not None)
    if ctx.torch_info is not None and not ctx.torch_info.get("cuda_available"):
        warnings.insert(0, "PyTorch non vede la GPU: il job girerà sulla CPU (molto lento)")
    serious = [w for w in warnings if not w.startswith("picco di VRAM non ancora")
               and not w.startswith("un altro job")]  # fmt: skip
    if not serious:
        if warnings:
            ctx.status_message.emit(" · ".join(warnings))
        return True
    box = QMessageBox(QMessageBox.Warning, "Controllo GPU", f"Prima di avviare {what}:",
                      parent=parent)  # fmt: skip
    box.setInformativeText("\n".join(f"• {w}" for w in warnings))
    box.addButton("Avvia comunque", QMessageBox.AcceptRole)
    cancel = box.addButton("Annulla", QMessageBox.RejectRole)
    box.exec()
    return box.clickedButton() is not cancel


def resume_job(parent: QWidget, ctx: AppContext, job: Job) -> bool:
    """Resume a paused/interrupted job, after checking what changed since the pause."""
    if job.command in ("train", "pretrain") and job.run_dir:
        state = read_state(job.run_dir) or {}
        changes = environment_changes(state)
        where = (f"epoca {state.get('epoch', 0) + 1}, batch {state.get('batch', 0)}"
                 if state.get("phase") == "train" else "valutazione finale")  # fmt: skip
        text = (f"Il run riprende da: {where}.\nLavoro già svolto: "
                f"{fmt_duration(state.get('active_s'))}.")  # fmt: skip
        if state.get("reason") == "auto":
            text += (
                "\nIl checkpoint è automatico (il job non era in pausa): il lavoro successivo "
                "all'ultimo checkpoint viene rifatto."
            )
        if changes:
            box = QMessageBox(QMessageBox.Warning, "Ambiente cambiato",
                              text + "\n\nDalla pausa sono cambiati:", parent=parent)  # fmt: skip
            box.setInformativeText("\n".join(f"• {c}" for c in changes)
                                   + "\n\nIl risultato potrebbe non essere identico a un run "
                                     "senza interruzioni.")  # fmt: skip
            box.addButton("Riprendi comunque", QMessageBox.AcceptRole)
            cancel = box.addButton("Annulla", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is cancel:
                return False
        else:
            ctx.status_message.emit(text.replace("\n", " "))
    if job.uses_gpu and not confirm_gpu_start(parent, ctx, None, f"«{job.title}»"):
        return False
    ctx.store.resume(job)
    ctx.tick()
    return True


def resume_run(parent: QWidget, ctx: AppContext, run_dir: str | Path) -> Job | None:
    """Create a job that resumes a run folder not known to the job list."""
    state = read_state(run_dir) or {}
    command = "pretrain" if state.get("kind") == "dino" else "train"
    job = ctx.store.add(command, ["--resume", str(run_dir)], f"Ripresa {Path(run_dir).name}",
                        run_dir=str(run_dir), state="paused")  # fmt: skip
    if resume_job(parent, ctx, job):
        return job
    ctx.store.remove(job)
    return None


def open_folder(path: str | Path) -> None:
    path = str(path)
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and \
            not os.environ.get("WAYLAND_DISPLAY"):  # fmt: skip
        return
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)) and sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
