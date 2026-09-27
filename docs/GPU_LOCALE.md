# Esecuzione locale su GPU NVIDIA RTX 4060 (8 GB)

Questa guida porta dall'installazione a un training completo su una singola RTX 4060 (desktop o laptop, 8 GB di VRAM).
Le configurazioni in `configs/` sono già dimensionate per questa scheda.

> ⚠️ **Stato della verifica.** Il codice è testato end-to-end su CPU (test automatici e run su dati sintetici).
> I valori di memoria GPU riportati sotto sono **stime**: prima di un run lungo misurali sulla tua scheda con
> `rsseg benchmark` (§4), che esegue passi di training reali e riporta il picco di VRAM.

## 1. Requisiti

| Componente | Versione |
|---|---|
| Sistema | Linux oppure Windows 11 con **WSL2** (consigliato). Su Windows nativo funziona, ma il caricamento dati multiprocesso è più lento. |
| Driver NVIDIA | Recente, compatibile con la build CUDA di PyTorch che installi (`nvidia-smi` mostra la versione CUDA massima supportata) |
| Python | 3.10–3.12 |
| Disco | ~60 GB per Potsdam grezzo + crop preparati |
| RAM | ≥ 16 GB (la valutazione su tile intere tiene in RAM una tile 6000² di probabilità, ~0.9 GB) |

## 2. Installazione

```bash
git clone <questo repository> && cd tesi-magistrale-cs-2022-matteo-campana
python -m venv .venv && source .venv/bin/activate        # oppure: uv venv && source .venv/bin/activate

# 1) PyTorch con CUDA: scegli l'indice adatto al tuo driver su https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) il pacchetto (Keras 3, MLflow, ...)
pip install -e ".[dev]"

# 3) verifica
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
pytest -q
```

Il backend Keras è impostato automaticamente a `torch` da `import rsseg` (`KERAS_BACKEND=torch`).

## 3. Prova rapida senza dataset (~2 minuti)

```bash
rsseg synth --out data/synthetic_raw --size 256
rsseg prepare -c configs/smoke.yaml
rsseg train   -c configs/smoke.yaml --set train.precision=mixed_bfloat16
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://127.0.0.1:5000
```

## 4. Dimensionare memoria e batch: `rsseg benchmark`

```bash
rsseg benchmark -c configs/experiments/maresunet_r18_imagenet.yaml
rsseg benchmark -c configs/experiments/maresunet_vq_r34_imagenet.yaml
rsseg benchmark -c configs/ssl/dino_r18.yaml --mode dino
```

L'output contiene `peak_gpu_memory_gb`, `seconds_per_step` e `images_per_second`. Tieni il picco sotto circa 7 GB
per lasciare margine al contesto CUDA e alla frammentazione.

### Leve per ridurre la memoria (in ordine di impatto)

| Leva | Come | Effetto |
|---|---|---|
| Batch più piccolo, stessa batch effettiva | `--set train.batch_size=4 train.grad_accum_steps=8` | Memoria di attivazione ≈ dimezzata, gradienti equivalenti a batch 32 (con BatchNorm su 4 campioni) |
| Crop più piccoli | `--set data.crop_size=384` (poi rifai `prepare`) | Memoria ∝ area del crop |
| Precisione | `mixed_bfloat16` (predefinita, supportata dalle RTX 40xx) | Circa −40% di attivazioni rispetto a float32. Non richiede loss scaling. |
| Backbone | `resnet18` < `resnet34` < `resnet50` | R50 difficilmente sta in 8 GB a 512² × 8 |
| DINO | `--set ssl.batch_size=16 ssl.local_crops_number=4` | Il costo cresce con batch × numero di viste |

### Configurazioni predefinite

| Uso | Configurazione | Stima VRAM |
|---|---|---|
| Segmentazione R18/R34 | crop 512, batch 8, accumulo 4 (effettiva 32), bf16 | ~4–6 GB |
| Segmentazione VQ | come sopra, codebook 512×512 | +0.2–0.5 GB |
| DINO R18 | batch 32, 2×224 + 6×96, head 16384 prototipi, bf16 | ~5–7 GB |
| Valutazione tile intere | finestre 512, batch 8 | ~1–2 GB |

## 5. Dataset reale

```bash
# scarica ISPRS Potsdam (2_Ortho_RGB e 5_Labels_all) accettando le condizioni ISPRS
rsseg prepare -c configs/base.yaml   # dati in data/download/Potsdam (vedi DATASET.md)
```

`prepare` usa `data.prepare_workers` processi. Ogni processo usa circa 1 GB di RAM per convertire una tile 6000²: con 16 GB di RAM tieni `prepare_workers` ≤ 4.

## 6. Training ed esperimenti

```bash
# baseline della tesi (modello migliore): MAResU-Net R18 ImageNet
rsseg train -c configs/experiments/maresunet_r18_imagenet.yaml

# pre-training DINO dell'encoder, poi fine-tuning
rsseg pretrain -c configs/ssl/dino_r18.yaml
rsseg train    -c configs/experiments/maresunet_r18_dino.yaml

# regime a poche etichette (RQ3)
rsseg train -c configs/experiments/maresunet_r18_dino.yaml --set data.label_fraction=0.1 name=r18_dino_10pct

# intera griglia della tesi su 3 seed
bash scripts/run_thesis_grid.sh
```

Tempi indicativi: un'epoca su Potsdam con crop 512 non sovrapposti (~3.000 crop di training con 22 tile) richiede
pochi minuti con R18 in bf16. Misura il tuo caso con `images_per_second` di `rsseg benchmark`.

## 7. MLflow

Ogni comando apre un run MLflow nell'esperimento `rsseg`. Per impostazione predefinita il tracking usa il file
`sqlite:///mlflow.db` nella directory corrente. Si può cambiare con `tracking.tracking_uri` o con la variabile
d'ambiente `MLFLOW_TRACKING_URI`.

| Cosa viene registrato | Dove |
|---|---|
| Configurazione completa (parametri appiattiti + `config.yaml`) | params, artifacts |
| Loss e metriche per epoca: `miou`, `mf1`, `oa`, `miou_no_clutter` e le versioni `val_*` | metrics |
| Learning rate, picco di VRAM (`gpu_peak_mem_gb`) | metrics |
| Perplexity e utilizzo dei codebook VQ (`vq_perplexity`, `vq_usage`) | metrics |
| DINO: `loss`, `teacher_entropy` (collasso se → 0), `lr`, `wd`, `momentum`, `teacher_temp` | metrics |
| Predizioni sul batch di riferimento (TOP \| GT \| pred \| errori) ogni N epoche | artifacts/predictions |
| Miglior checkpoint (`best.weights.h5`) ed elenco dei top-5 (`topk.json`), backbone DINO. I 5 checkpoint migliori restano solo in `runs/<run>/checkpoints/` | artifacts/model, artifacts/backbone |
| Epoche effettivamente eseguite (`epochs_trained`: < `train.epochs` se è scattato l'early stopping) | metrics |
| Valutazione su tile intere: `report.json`, `report.md`, mappe di predizione ed errore | artifacts/evaluation |
| Ambiente: versioni, GPU, commit git | tags `env.*` |

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 8. Problemi comuni

| Sintomo | Causa probabile e soluzione |
|---|---|
| `torch.cuda.is_available() == False` | Build CPU di PyTorch installata: reinstalla con l'indice CUDA (§2) |
| `CUDA out of memory` | Vedi §4. Chiudi altre applicazioni che usano la GPU (browser con accelerazione hardware). |
| Warning "falling back to mixed_float16" | GPU senza bf16. Con le RTX 40xx non dovrebbe succedere: controlla driver e versione di PyTorch. |
| DataLoader lento su Windows | Usa WSL2, oppure `--set data.num_workers=2` |
| `teacher_entropy` → 0 in DINO | Collasso: riduci il learning rate, aumenta `ssl.warmup_teacher_temp_epochs`, oppure parti da `ssl.weights=imagenet` |
| `vq_usage` molto basso | Pochi codici usati: il decoder aggira il bottleneck (RQ2). Prova `model.vq.on_skips=true` o un `num_codes` più piccolo |
