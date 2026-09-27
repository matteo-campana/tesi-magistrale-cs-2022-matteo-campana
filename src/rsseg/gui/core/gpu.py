# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU status: detection, utilisation, memory, temperature and processes.

The application samples the GPU about once a second without importing torch:

* **NVML** (``nvidia-ml-py``, imported as ``pynvml``) when available — microseconds per sample;
* otherwise ``nvidia-smi --query-gpu=... --format=csv`` (tens of milliseconds);
* otherwise no NVIDIA GPU is reported, and the application stays usable (e.g. for results).

A working driver is not enough: PyTorch installed without CUDA would train on the CPU without
any error. :data:`TORCH_CHECK` is run once in a separate process to tell the two apart
(:func:`status`).
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

TORCH_CHECK = (
    "import json, torch\n"
    "cuda = torch.cuda.is_available()\n"
    "print(json.dumps({'torch': torch.__version__, 'cuda_available': cuda,\n"
    "  'cuda_version': torch.version.cuda,\n"
    "  'device': torch.cuda.get_device_name(0) if cuda else None,\n"
    "  'bf16': bool(cuda and torch.cuda.is_bf16_supported()),\n"
    "  'capability': list(torch.cuda.get_device_capability(0)) if cuda else None}))\n"
)
"""Program printing what PyTorch sees (run it with the Python of the project)."""

SMI_FIELDS = ("index", "name", "utilization.gpu", "memory.used", "memory.total",
              "temperature.gpu", "power.draw", "power.limit", "clocks.sm", "fan.speed",
              "driver_version")  # fmt: skip

MEMORY_ALERT = 0.95
TEMP_ALERT_C = 85.0


@dataclass
class GPUProcess:
    pid: int
    name: str
    memory_gb: float | None


@dataclass
class GPUSample:
    time: float
    index: int
    name: str
    util: float | None
    """Compute utilisation, %."""
    mem_used_gb: float | None
    mem_total_gb: float | None
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    clock_mhz: float | None = None
    fan_pct: float | None = None
    processes: list[GPUProcess] = field(default_factory=list)

    @property
    def mem_free_gb(self) -> float | None:
        if self.mem_used_gb is None or self.mem_total_gb is None:
            return None
        return self.mem_total_gb - self.mem_used_gb

    @property
    def mem_fraction(self) -> float | None:
        if self.mem_used_gb is None or not self.mem_total_gb:
            return None
        return self.mem_used_gb / self.mem_total_gb

    def as_dict(self) -> dict:
        return asdict(self)


def _num(text: str) -> float | None:
    text = text.strip().split(" ")[0]
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


def parse_smi(text: str, now: float | None = None) -> tuple[list[GPUSample], str | None]:
    """Samples from ``nvidia-smi --query-gpu=<SMI_FIELDS> --format=csv,noheader,nounits``."""
    samples, driver = [], None
    for row in csv.reader(io.StringIO(text)):
        if len(row) < len(SMI_FIELDS):
            continue
        v = dict(zip(SMI_FIELDS, (c.strip() for c in row), strict=False))
        mib = 1024.0
        used, total = _num(v["memory.used"]), _num(v["memory.total"])
        samples.append(GPUSample(
            time=now or time.time(), index=int(_num(v["index"]) or 0), name=v["name"],
            util=_num(v["utilization.gpu"]),
            mem_used_gb=used / mib if used is not None else None,
            mem_total_gb=total / mib if total is not None else None,
            temp_c=_num(v["temperature.gpu"]), power_w=_num(v["power.draw"]),
            power_limit_w=_num(v["power.limit"]), clock_mhz=_num(v["clocks.sm"]),
            fan_pct=_num(v["fan.speed"]),
        ))  # fmt: skip
        driver = v["driver_version"]
    return samples, driver


def _process_name(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(pid).name()
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "?"


class GPUProbe:
    """Samples the NVIDIA GPUs through NVML or ``nvidia-smi`` (``mode``: nvml, smi, none)."""

    def __init__(self, prefer: str = "nvml"):
        self.mode = "none"
        self.driver: str | None = None
        self.cuda_driver: str | None = None
        self.error: str | None = None
        self._nvml = None
        if prefer == "nvml" and self._init_nvml():
            self.mode = "nvml"
        elif shutil.which("nvidia-smi"):
            self.mode = "smi"

    def _init_nvml(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self.driver = _text(pynvml.nvmlSystemGetDriverVersion())
            v = pynvml.nvmlSystemGetCudaDriverVersion()
            self.cuda_driver = f"{v // 1000}.{v % 1000 // 10}"
            return True
        except Exception as e:  # noqa: BLE001 - no library, no driver, no GPU
            self.error = str(e) or type(e).__name__
            return False

    def sample(self) -> list[GPUSample]:
        try:
            if self.mode == "nvml":
                return self._sample_nvml()
            if self.mode == "smi":
                return self._sample_smi()
        except Exception as e:  # noqa: BLE001 - driver reset, GPU lost
            self.error = str(e) or type(e).__name__
        return []

    def _sample_nvml(self) -> list[GPUSample]:
        nv, now, out = self._nvml, time.time(), []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            mem = nv.nvmlDeviceGetMemoryInfo(h)

            def opt(fn, *a, scale=1.0, h=h):
                try:
                    return fn(h, *a) / scale
                except Exception:  # noqa: BLE001 - not supported on this card
                    return None

            procs = []
            try:
                for pr in nv.nvmlDeviceGetComputeRunningProcesses(h):
                    used = getattr(pr, "usedGpuMemory", None)
                    procs.append(GPUProcess(pr.pid, _process_name(pr.pid),
                                            used / 2**30 if used else None))  # fmt: skip
            except Exception:  # noqa: BLE001
                pass
            util = opt(lambda h: nv.nvmlDeviceGetUtilizationRates(h).gpu)
            out.append(GPUSample(
                time=now, index=i, name=_text(nv.nvmlDeviceGetName(h)), util=util,
                mem_used_gb=mem.used / 2**30, mem_total_gb=mem.total / 2**30,
                temp_c=opt(nv.nvmlDeviceGetTemperature, nv.NVML_TEMPERATURE_GPU),
                power_w=opt(nv.nvmlDeviceGetPowerUsage, scale=1000.0),
                power_limit_w=opt(nv.nvmlDeviceGetEnforcedPowerLimit, scale=1000.0),
                clock_mhz=opt(nv.nvmlDeviceGetClockInfo, nv.NVML_CLOCK_SM),
                fan_pct=opt(nv.nvmlDeviceGetFanSpeed), processes=procs,
            ))  # fmt: skip
        return out

    def _sample_smi(self) -> list[GPUSample]:
        text = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout  # fmt: skip
        samples, self.driver = parse_smi(text)
        return samples


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def check_torch(python: str | None = None, timeout: float = 120) -> dict:
    """Run :data:`TORCH_CHECK` in a separate process (importing torch takes seconds)."""
    try:
        r = subprocess.run([python or sys.executable, "-c", TORCH_CHECK], capture_output=True,
                           text=True, timeout=timeout)  # fmt: skip
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    if r.returncode:
        return {"error": (r.stderr.strip().splitlines() or ["torch not importable"])[-1]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": r.stdout.strip() or "no output"}


def status(samples: list[GPUSample], torch_info: dict | None) -> tuple[str, str]:
    """``(level, message)``: ``ready`` (green), ``warning`` (yellow) or ``absent`` (red)."""
    if not samples:
        if torch_info and torch_info.get("cuda_available"):
            return "ready", f"GPU pronta: {torch_info.get('device')} (monitoraggio non disponibile)"
        return "absent", ("GPU NVIDIA non rilevata: il training userebbe la CPU (molto lento). "
                          "Controlla il driver con `nvidia-smi`.")  # fmt: skip
    name = samples[0].name
    if torch_info is None:
        return "warning", f"{name} rilevata — verifica di PyTorch in corso…"
    if torch_info.get("error"):
        return "warning", f"{name} rilevata, ma PyTorch non si importa: {torch_info['error']}"
    if not torch_info.get("cuda_available"):
        return "warning", (
            f"{name} rilevata, ma PyTorch {torch_info.get('torch')} è senza CUDA: il training "
            "userebbe la CPU. Reinstalla PyTorch con CUDA, ad esempio\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        )
    extra = "" if torch_info.get("bf16") else " (senza bfloat16: si userà float16)"
    return "ready", f"GPU pronta: {name} · CUDA {torch_info.get('cuda_version')}{extra}"


def alerts(sample: GPUSample) -> list[str]:
    out = []
    if (f := sample.mem_fraction) is not None and f >= MEMORY_ALERT:
        out.append(f"memoria GPU quasi piena ({f:.0%}): rischio di out-of-memory")
    if sample.temp_c is not None and sample.temp_c >= TEMP_ALERT_C:
        out.append(f"temperatura GPU alta ({sample.temp_c:.0f} °C)")
    return out


def preflight(sample: GPUSample | None, needed_gb: float | None, own_pids: set[int],
              gpu_job_running: bool) -> list[str]:  # fmt: skip
    """Warnings before starting a GPU job (empty: nothing to report)."""
    if sample is None:
        return ["nessuna GPU rilevata: il job girerà sulla CPU"]
    out = []
    if gpu_job_running:
        out.append("un altro job usa già la GPU: questo resterà in coda fino alla sua fine")
    others = [p for p in sample.processes if p.pid not in own_pids]
    busy = sum(p.memory_gb or 0 for p in others)
    if busy > 0.5:
        names = ", ".join(sorted({p.name for p in others}))
        out.append(f"altri processi occupano {busy:.1f} GB di VRAM ({names})")
    free = sample.mem_free_gb
    if needed_gb is not None and free is not None and needed_gb > free:
        out.append(f"servono ~{needed_gb:.1f} GB di VRAM (misurati con il benchmark), "
                   f"liberi {free:.1f} GB")  # fmt: skip
    elif needed_gb is None:
        out.append("picco di VRAM non ancora misurato: esegui «Entra in 8 GB?» (benchmark)")
    return out


class GPULog:
    """Appends samples to a CSV file (``gpu_log.csv`` of a job)."""

    COLUMNS = ("time", "index", "util", "mem_used_gb", "mem_total_gb", "temp_c", "power_w")

    def __init__(self, path):
        self.path = path

    def append(self, sample: GPUSample) -> None:
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(self.COLUMNS)
            w.writerow([getattr(sample, c) if getattr(sample, c) is not None else ""
                        for c in self.COLUMNS])  # fmt: skip

    def read(self) -> list[dict]:
        try:
            with self.path.open(newline="", encoding="utf-8") as f:
                return [
                    {k: float(v) if v else None for k, v in r.items()} for r in csv.DictReader(f)
                ]
        except OSError:
            return []
