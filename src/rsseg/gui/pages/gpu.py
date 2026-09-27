# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU: detection, PyTorch check, live utilisation, memory, temperature and processes."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rsseg.gui.context import AppContext
from rsseg.gui.core import gpu as gpucore
from rsseg.gui.widgets import LEVEL_COLORS, plot_series, plot_widget, section_title, tip

MODES = {"nvml": "NVML (nvidia-ml-py)", "smi": "nvidia-smi", "none": "non disponibile"}


class GPUPage(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        v = QVBoxLayout(self)
        v.addWidget(section_title("GPU"))

        status = QGroupBox("Stato")
        sv = QVBoxLayout(status)
        self.status = QLabel("Rilevamento…")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sv.addWidget(self.status)
        form = QFormLayout()
        self.driver = QLabel("—")
        self.mode = QLabel("—")
        self.torch = QLabel("verifica in corso…")
        tip(self.torch, "gpu", "torch")
        form.addRow("Driver NVIDIA:", self.driver)
        form.addRow("Monitoraggio:", self.mode)
        form.addRow("PyTorch:", self.torch)
        sv.addLayout(form)
        recheck = QPushButton("Riverifica GPU e PyTorch")
        recheck.setToolTip("Ripete il controllo di PyTorch (utile dopo aver reinstallato torch o "
                           "aggiornato il driver)")  # fmt: skip
        recheck.clicked.connect(ctx.check_torch)
        sv.addWidget(recheck, alignment=Qt.AlignLeft)
        v.addWidget(status)

        live = QGroupBox("Utilizzo in tempo reale")
        grid = QGridLayout(live)
        self.values: dict[str, QLabel] = {}
        items = [("Nome", "name", None), ("Utilizzo", "util", "util"),
                 ("Memoria", "memory", "memory"), ("Temperatura", "temp", "temperature"),
                 ("Potenza", "power", "power"), ("Frequenza SM", "clock", "clock"),
                 ("Ventola", "fan", None)]  # fmt: skip
        big = QFont()
        big.setPointSize(big.pointSize() + 2)
        for i, (label, key, help_key) in enumerate(items):
            name = QLabel(label + ":")
            value = QLabel("—")
            value.setFont(big)
            if help_key:
                tip(name, "gpu", help_key)
                tip(value, "gpu", help_key)
            grid.addWidget(name, i // 2, (i % 2) * 2)
            grid.addWidget(value, i // 2, (i % 2) * 2 + 1)
            self.values[key] = value
        self.mem_bar = QProgressBar()
        self.mem_bar.setRange(0, 1000)
        tip(self.mem_bar, "gpu", "memory")
        grid.addWidget(self.mem_bar, 4, 0, 1, 4)
        self.alerts = QLabel()
        self.alerts.setStyleSheet("color:#c0392b; font-weight:bold;")
        grid.addWidget(self.alerts, 5, 0, 1, 4)
        v.addWidget(live)

        charts = QHBoxLayout()
        self.util_plot = plot_widget("Utilizzo %", "%", "secondi fa")
        self.mem_plot = plot_widget("Memoria", "GB", "secondi fa")
        self.temp_plot = plot_widget("Temperatura", "°C", "secondi fa")
        for p in (self.util_plot, self.mem_plot, self.temp_plot):
            p.setMinimumHeight(180)
            charts.addWidget(p)
        v.addLayout(charts)

        v.addWidget(tip(QLabel("<b>Processi sulla GPU</b>"), "gpu", "processes"))
        self.procs = QTableWidget(0, 3)
        self.procs.setHorizontalHeaderLabels(["PID", "Processo", "Memoria"])
        self.procs.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.procs.verticalHeader().hide()
        self.procs.setEditTriggers(QTableWidget.NoEditTriggers)
        self.procs.setMaximumHeight(160)
        v.addWidget(self.procs)

        ctx.gpu_sampled.connect(self.on_samples)
        ctx.gpu_ready.connect(self.on_ready)
        ctx.torch_checked.connect(lambda _: self.refresh_status())
        self._tick = 0

    def showEvent(self, event):  # noqa: N802 (Qt API)
        super().showEvent(event)
        self.on_samples(self.ctx.samples)

    def on_ready(self) -> None:
        self.driver.setText(" · ".join(x for x in (
            self.ctx.driver and f"versione {self.ctx.driver}",
            self.ctx.cuda_driver and f"CUDA supportata fino a {self.ctx.cuda_driver}") if x)
            or "non rilevato")  # fmt: skip
        self.mode.setText(MODES.get(self.ctx.gpu_mode, self.ctx.gpu_mode))
        self.refresh_status()

    def refresh_status(self) -> None:
        level, message = gpucore.status(self.ctx.samples, self.ctx.torch_info)
        self.status.setText(f"<span style='color:{LEVEL_COLORS[level]}; font-size:18px'>●</span> "
                            + message.replace("\n", "<br>"))  # fmt: skip
        t = self.ctx.torch_info
        if t is None:
            self.torch.setText("verifica in corso…")
        elif t.get("error"):
            self.torch.setText(f"errore: {t['error']}")
        else:
            self.torch.setText(
                f"{t.get('torch')} · CUDA {'sì' if t.get('cuda_available') else 'NO'}"
                + (f" ({t.get('cuda_version')}) · {t.get('device')} · bfloat16 "
                   f"{'sì' if t.get('bf16') else 'no'}" if t.get("cuda_available") else "")
            )  # fmt: skip

    def on_samples(self, samples) -> None:
        self._tick += 1
        if self._tick % 5 == 1:
            self.refresh_status()
        if not samples:
            for w in self.values.values():
                w.setText("—")
            self.alerts.setText("")
            return
        s = samples[0]
        f = lambda x, fmt: "—" if x is None else fmt.format(x)  # noqa: E731
        self.values["name"].setText(s.name)
        self.values["util"].setText(f(s.util, "{:.0f} %"))
        self.values["memory"].setText(
            "—" if s.mem_used_gb is None else f"{s.mem_used_gb:.2f} / {s.mem_total_gb:.2f} GB "
            f"(libera {s.mem_free_gb:.2f} GB)")  # fmt: skip
        self.values["temp"].setText(f(s.temp_c, "{:.0f} °C"))
        power = f(s.power_w, "{:.0f} W")
        if s.power_limit_w:
            power += f" / {s.power_limit_w:.0f} W"
        self.values["power"].setText(power)
        self.values["clock"].setText(f(s.clock_mhz, "{:.0f} MHz"))
        self.values["fan"].setText(f(s.fan_pct, "{:.0f} %"))
        if s.mem_fraction is not None:
            self.mem_bar.setValue(int(s.mem_fraction * 1000))
            self.mem_bar.setFormat(f"VRAM {s.mem_fraction:.0%}")
        self.alerts.setText("  ·  ".join(gpucore.alerts(s)))
        if not self.isVisible():
            return
        hist = list(self.ctx.gpu_history)
        if hist:
            now = hist[-1].time
            x = [h.time - now for h in hist]
            plot_series(self.util_plot, {"utilizzo": (x, [h.util for h in hist])})
            plot_series(self.mem_plot, {"usata": (x, [h.mem_used_gb for h in hist]),
                                        "totale": (x, [h.mem_total_gb for h in hist])})  # fmt: skip
            plot_series(self.temp_plot, {"temperatura": (x, [h.temp_c for h in hist])})
        own = self.ctx.own_pids()
        self.procs.setRowCount(len(s.processes))
        for row, p in enumerate(s.processes):
            mine = p.pid in own
            cells = [str(p.pid), p.name + ("  (job rsseg)" if mine else ""),
                     "—" if p.memory_gb is None else f"{p.memory_gb:.2f} GB"]  # fmt: skip
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if mine:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.procs.setItem(row, col, item)
