# Esecuzione locale su GPU NVIDIA RTX 4060 (8 GB)

Questa guida porta dall'installazione a un training completo su una singola RTX 4060 (desktop o laptop, 8 GB di VRAM).
Le configurazioni in `configs/` sono già dimensionate per questa scheda.

> **Stato della verifica.** I valori di memoria e velocità riportati sotto sono **misurati** su una RTX 4060 da
> 8 GB (Windows 11 nativo, PyTorch 2.14 + CUDA 13.2) con `rsseg benchmark` e con training brevi sui dati reali.
> Su un'altra scheda misurali con `rsseg benchmark` (§4). I tempi di test ed esperimenti sono in
> [ESECUZIONE_E_TEMPI.md](ESECUZIONE_E_TEMPI.md).

## 1. Requisiti

| Componente | Versione |
|---|---|
| Sistema | Linux oppure Windows 11 con **WSL2** (consigliato). Su Windows nativo funziona, ma il caricamento dati multiprocesso è più lento. |
| Driver NVIDIA | Recente, compatibile con la build CUDA di PyTorch che installi (`nvidia-smi` mostra la versione CUDA massima supportata) |
| Python | 3.10–3.12 |
| Disco | ~60 GB per i dati ISPRS scaricati (Potsdam 31 GB, Vaihingen 23 GB, Toronto 5 GB) + ~6 GB di dati preparati + fino a ~200 GB di checkpoint per tutte le campagne |
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
| Backbone | `resnet18` < `resnet34` < `resnet50` | R50 **non** sta in 8 GB a 512² × 8 (misurato): usa batch 4 e accumulo 8 |
| DINO | `--set ssl.batch_size=16 ssl.local_crops_number=4` | Il costo cresce con batch × numero di viste |

### Configurazioni predefinite

Picco di VRAM misurato su RTX 4060 (`rsseg benchmark`, campagna `rq5_efficiency`):

| Uso | Configurazione | VRAM | Immagini/s (solo GPU) |
|---|---|---|---|
| MAResU-Net R18 | crop 512, batch 8, accumulo 4 (effettiva 32), bf16 | 4,8 GB | 21 |
| MAResU-Net R18 + nDSM | come sopra, 4 canali | 4,8 GB | 23 |
| MAResU-Net-VQ R18 | come sopra, codebook 512 | 4,7 GB | 23 |
| MAResU-Net R34 | come sopra | 5,5 GB | 19 |
| MAResU-Net R50 | batch 8 | **OOM** | — |
| MAResU-Net Swin-T | batch 4, accumulo 8 | 4,2 GB | 13 |
| MAResU-Net ViT-B/16 | batch 4, accumulo 8 | 7,1 GB | 2 |
| DINO R18 | batch 32, 2×224 + 6×96, head 16384 prototipi, bf16 | 4,2 GB | 85 |
| DINO R34 | come sopra | 6,2 GB | 40 |

Con la GPU condivisa con altri programmi (browser, IDE) il margine si riduce: se un run va in OOM, chiudili o
usa le leve della tabella precedente.

## 5. Dataset reale

```bash
# scarica e decomprimi i dati ISPRS in data/download/{Potsdam,Vaihingen,Toronto} (vedi DATASET.md)
rsseg prepare -c configs/experiments/maresunet_r18_imagenet_ndsm.yaml --dry-run   # file scelti e anomalie
rsseg prepare -c configs/experiments/maresunet_r18_imagenet_ndsm.yaml             # Potsdam con nDSM (~3 min)
rsseg prepare -c configs/experiments/vaihingen_maresunet_r18_imagenet_ndsm.yaml   # Vaihingen (~30 s)
```

Le campagne (§6) preparano da sole i dati che mancano. `prepare` usa `data.prepare_workers` processi. Ogni
processo usa circa 1 GB di RAM per convertire una tile 6000²: con 16 GB di RAM tieni `prepare_workers` ≤ 4.

## 6. Training ed esperimenti

```bash
# baseline della tesi (modello migliore): MAResU-Net R18 ImageNet
rsseg train -c configs/experiments/maresunet_r18_imagenet.yaml

# pre-training DINO dell'encoder, poi fine-tuning
rsseg pretrain -c configs/ssl/dino_r18.yaml
rsseg train    -c configs/experiments/maresunet_r18_dino.yaml

# regime a poche etichette (RQ3)
rsseg train -c configs/experiments/maresunet_r18_dino.yaml --set data.label_fraction=0.1 name=r18_dino_10pct
```

### Campagne: tutti gli esperimenti con un comando

Le campagne (`configs/campaigns/*.yaml`) elencano esperimenti, seed, sweep, backbone DINO e pool non etichettati
di una domanda di ricerca. `rsseg campaign` le esegue un passo alla volta, ognuno in un sottoprocesso:

```bash
rsseg campaign configs/campaigns/all.yaml --dry-run          # cosa verrebbe eseguito, disco e tempo stimati
rsseg campaign configs/campaigns/rq6_ndsm.yaml --quick       # prova veloce (1 epoca, 1 tile) in runs/campaigns-quick/
rsseg campaign configs/campaigns/rq6_ndsm.yaml               # esecuzione vera; rilanciata, riprende da dove era
rsseg campaign configs/campaigns/all.yaml --only "vaihingen_*" --seeds 0   # sottoinsieme
```

| Campagna | Contenuto |
|---|---|
| `thesis_grid` | griglia della tesi: MAResU-Net/VQ × R18/R34 × ImageNet/DINO (anche `bash scripts/run_thesis_grid.sh`) |
| `rq1_attention` | MAResU-Net vs U-Net |
| `rq3_low_label` | 1/10/25/100% delle etichette × ImageNet, DINO, DINO + Toronto |
| `rq4_aerial` | DINO con augmentation standard vs "aerial" |
| `rq5_efficiency` | solo `rsseg benchmark`: parametri, immagini/s, VRAM |
| `rq6_ndsm` | RGB vs RGB + nDSM (ufficiale `lastools`, calcolato dal DSM) |
| `vaihingen` | RQ1 e RQ6 su Vaihingen |
| `transformers` | encoder Swin-T, ViT-B/16, ViT-S/16 (DINO) |
| `thesis2022` | protocollo della tesi, per il confronto con la tab. 4.5 |
| `all` | tutte le precedenti |

Come funziona:

- **Prerequisiti**: se mancano, prepara i dati (`rsseg prepare`, dopo un `--dry-run` che deve essere senza errori)
  e i pool non etichettati. Un `data.root` già preparato con impostazioni diverse (crop, split, nDSM) blocca la
  campagna invece di essere sovrascritto.
- **Store condiviso**: ogni passo vive in `runs/campaigns/store/{backbones,runs,benchmarks}/<id>/` con `spec.json`,
  `log.txt` e `outcome.json`. I passi completati vengono saltati: una campagna interrotta riprende da dove era, e un
  run comune a più campagne (ad esempio la baseline `maresunet_r18_imagenet`) viene addestrato una volta sola.
  Due esperimenti diversi con lo stesso id vengono rifiutati.
- **Errori**: un passo fallito (es. memoria esaurita) viene registrato e la campagna prosegue; i run che dipendono
  da un pre-training fallito risultano `blocked`. `--stop-on-error` ferma tutto al primo errore, `--clean-failed`
  cancella le cartelle dei tentativi falliti prima di riprovare.
- **Avanzamento**: durante ogni passo la console mostra epoca, batch, `val_mIoU` e migliore, epoche rimaste prima
  dell'early stopping, tile in valutazione, tempo rimasto del passo e della campagna. Dettagli in
  [ESECUZIONE_E_TEMPI.md](ESECUZIONE_E_TEMPI.md#44-seguire-una-campagna-in-corso).
- **Risultati**: `runs/campaigns/<campagna>/report.md` (media ± std sui seed, metriche piene ed eroded, benchmark,
  passi falliti con il log) e `status.json`. Contiene solo i run della campagna, mai run estranei.
- **Disco**: prima di partire stima lo spazio (checkpoint top-k con lo stato dell'ottimizzatore, circa 260 MB
  l'uno per R18) e si ferma se non basta (`--ignore-disk` per forzare). `all.yaml` richiede circa 200 GB.
- Il pre-training DINO è condiviso da tutti i seed del fine-tuning, salvo `per_seed: true`: il report lo dichiara.

Tempi misurati per epoca, per run e per campagna: [ESECUZIONE_E_TEMPI.md](ESECUZIONE_E_TEMPI.md).

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
| Uno script Python tuo si blocca all'avvio del DataLoader su Windows | I worker rieseguono lo script: metti il codice sotto `if __name__ == "__main__":` (i comandi `rsseg` lo fanno già) |
| Una campagna segna un passo `failed` | Leggi il `log.txt` indicato nel report; correggi e rilancia la stessa campagna: riparte dai passi non completati |
| `teacher_entropy` → 0 in DINO | Collasso: riduci il learning rate, aumenta `ssl.warmup_teacher_temp_epochs`, oppure parti da `ssl.weights=imagenet` |
| `vq_usage` molto basso | Pochi codici usati: il decoder aggira il bottleneck (RQ2). Prova `model.vq.on_skips=true` o un `num_codes` più piccolo |
