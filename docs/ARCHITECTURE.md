# Architettura

Il documento descrive il pacchetto `rsseg` (*Remote Sensing SEGmentation*), implementato con **Keras 3 su
backend PyTorch** e con tracking degli esperimenti su **MLflow**. Le motivazioni delle scelte sono in
[ANALISI_TESI.md](ANALISI_TESI.md) e [REVISIONE_CODICE_ORIGINALE.md](REVISIONE_CODICE_ORIGINALE.md).
L'esecuzione su GPU locale è descritta in [GPU_LOCALE.md](GPU_LOCALE.md).

## 1. Obiettivi e vincoli

| Obiettivo | Conseguenza architetturale |
|---|---|
| Ritestare le ipotesi della tesi (LAM, VQ, DINO) in modo pulito | Parti ortogonali e combinabili da configurazione: `backbone × skip × neck × pre-training` |
| Riproducibilità | Seed globale, config salvata con ogni run, split deterministici per tile, manifest dei dati, commit git registrato su MLflow |
| Confrontabilità con la letteratura | Split standard di Potsdam, valutazione su tile intere, matrice di confusione globale |
| Singola GPU da 8 GB (RTX 4060) | bf16 mixed precision, accumulo del gradiente, crop 512, `rsseg benchmark` per misurare la VRAM |
| Manutenibilità | Pacchetto installabile, test automatici, ruff, CI |

### Stack

| Livello | Scelta | Perché |
|---|---|---|
| Modelli e training | **Keras 3** (`keras.Model`, `fit`, callback) | API ad alto livello richiesta. Modelli funzionali leggibili, salvataggio dei pesi `.weights.h5`. |
| Backend | **PyTorch** (`KERAS_BACKEND=torch`, impostato da `import rsseg`) | CUDA maturo sulle GPU consumer, pesi ImageNet di torchvision, `DataLoader` accettato direttamente da `fit` |
| Tracking | **MLflow** (default `sqlite:///mlflow.db`) | Parametri, metriche per epoca, artefatti (checkpoint, report, figure), confronto tra run |
| Dati | PyTorch `Dataset`/`DataLoader`, Pillow, NumPy | Caricamento parallelo con i worker. Nessuna dipendenza da TensorFlow. |
| Configurazione | Dataclass tipizzate + YAML componibili (`base:`) + override `--set k=v` | Nessuna dipendenza pesante, chiavi sconosciute rifiutate |

## 2. Vista d'insieme

```mermaid
flowchart LR
    subgraph data[rsseg.data]
        RAW[(ISPRS Potsdam<br/>tile 6000²)] --> PREP[prepare: split per tile,<br/>LUT RGB→id, crop]
        PREP --> MAN[(crops.csv · tiles.csv<br/>stats.json · split.json)]
        MAN --> SEGDL[CropDataset + JointAugment]
        MAN --> SSLDL[MultiCropDataset<br/>2×224 + N×96]
    end

    subgraph ssl[rsseg.ssl]
        SSLDL --> DINO[DINOModel<br/>student/teacher EMA<br/>train_step PyTorch]
        DINO -->|backbone.weights.h5| BB
    end

    subgraph models[rsseg.models]
        BB[ResNet18/34/50<br/>pesi torchvision] --> SKIP[skip: LAM+CAM or plain]
        SKIP --> NECK[neck: identity or VQ]
        NECK --> DEC[decoder MAResU-Net<br/>+ head float32]
    end

    SEGDL --> FIT[keras fit<br/>MaskedCE + VQ loss<br/>mIoU/mF1/OA]
    DEC --- FIT
    FIT --> EVAL[rsseg.eval<br/>sliding window su tile]
    FIT & DINO & EVAL --> MLF[(MLflow)]
```

## 3. Struttura del repository

```
.
├── LICENSE  README.md  CITATION.cff  pyproject.toml
├── configs/
│   ├── base.yaml                 # default dimensionati per GPU da 8 GB
│   ├── smoke.yaml                # run end-to-end su dati sintetici (CPU)
│   ├── experiments/              # griglia della tesi + baseline U-Net + protocollo thesis2022
│   └── ssl/                      # DINO R18/R34
├── src/rsseg/
│   ├── __init__.py               # imposta KERAS_BACKEND=torch
│   ├── config.py                 # dataclass + YAML + override
│   ├── runtime.py                # precisione (bf16/fp16/fp32), seed, flag CUDA
│   ├── tracking.py               # run MLflow, parametri, ambiente, callback metriche/VRAM
│   ├── cli.py                    # rsseg synth|prepare|train|pretrain|evaluate|benchmark|report
│   ├── data/
│   │   ├── classes.py            # classi, palette, LUT vettorizzata, mapping legacy
│   │   ├── splits.py             # protocolli standard / thesis2022
│   │   ├── prepare.py            # tile → crop + maschere di tile + manifest + statistiche
│   │   ├── datasets.py           # CropDataset, MultiCropDataset, label_fraction per tile
│   │   ├── transforms.py         # augment congiunte, multi-crop DINO (anche "aerial")
│   │   └── synthetic.py          # tile sintetiche con gli stessi nomi di file di Potsdam
│   ├── models/
│   │   ├── backbones.py          # ResNet Keras con nomi layer = torchvision
│   │   ├── torchvision_weights.py# porting pesi ImageNet da torchvision
│   │   ├── attention.py          # LinearPAM (LAM), ChannelAttention, blocco PAM/CAM
│   │   ├── quantizer.py          # VectorQuantizer (EMA in variabili, init dai dati, restart)
│   │   └── segmentor.py          # build_segmentor(ModelConfig)
│   ├── ssl/
│   │   ├── dino.py               # head, loss per campione, schedule, DINOModel.train_step
│   │   └── train.py              # pretrain_dino + monitor di collasso + export backbone
│   ├── train/
│   │   ├── losses.py             # MaskedSparseCrossEntropy (ignora solo 255)
│   │   ├── metrics.py            # mIoU / mF1 / OA da matrice di confusione globale
│   │   ├── callbacks.py          # CodebookMonitor, PredictionLogger
│   │   └── seg_train.py          # train_segmentation, evaluate_segmentation
│   └── eval/
│       ├── sliding_window.py     # inferenza su tile intere con finestre sovrapposte
│       ├── report.py             # report JSON/MD, aggregazione media ± std
│       ├── visualize.py          # pannelli TOP | GT | pred | errori
│       └── benchmark.py          # picco VRAM e throughput di una configurazione
├── scripts/run_thesis_grid.sh    # griglia 2×2×2 × seed
├── tests/                        # 32 test (unit + end-to-end su dati sintetici)
├── docs/
└── legacy/                       # materiale originale 2022, invariato
```

## 4. Componenti

### 4.1 Dati (`rsseg.data`)
- **`classes.py`** è l'unica fonte di verità per le classi. L'ordine è quello ISPRS/MMSegmentation, con **clutter per
  ultimo (indice 5)**. I colori sconosciuti diventano `255` (ignorato). La conversione RGB→id avviene con
  `searchsorted` su codici a 24 bit, senza loop per pixel.
- **`prepare.py`** assegna prima le **tile** agli split e poi taglia i crop, così nessun crop attraversa due split.
  La sovrapposizione (`train_stride < crop_size`) è ammessa solo nel train. Le tile di val/test vengono
  registrate intere per la valutazione a finestra scorrevole. Le statistiche di normalizzazione sono calcolate
  solo sul train.
- **Normalizzazione dentro il modello.** I dataset restituiscono immagini `float32` in `[0, 255]`, channels-last, e il
  primo layer del backbone applica media e deviazione standard ImageNet. L'inferenza quindi non può dimenticare
  la normalizzazione (bug S4 del 2022).
- **`label_fraction`** campiona i crop etichettati **per tile** con seed fisso, per il regime a poche etichette (RQ3).

### 4.2 Modelli (`rsseg.models`)

```
image ─► Backbone ─► [C1, C2, C3, C4] ─► skip(Ci) ─► Neck(C4) ─► Decoder(+ skip) ─► Head ─► logits (B, H, W, 6) float32
```

| Parte | Varianti | Note |
|---|---|---|
| Backbone | `resnet18`, `resnet34`, `resnet50` | Scritto in Keras (in `keras.applications` mancano R18/R34). Pesi `imagenet` portati da torchvision con equivalenza numerica testata, `none`, oppure un file `.weights.h5` (per esempio il backbone DINO). |
| Skip | `lam` (MAResU-Net: conv → LinearPAM ‖ ChannelAttention → 1×1), `plain` (identità) | `gamma` inizializzato a 0. Matematica dell'attenzione in float32. |
| Neck | `identity`, `vq` | `VectorQuantizer`: straight-through, commitment β, EMA oppure codebook appreso per gradiente, init dal primo batch, restart dei codici morti, `perplexity`/`usage`. Proiezione 1×1 se `code_dim` è diverso dai canali. Con `vq.on_skips=true` si quantizzano anche le skip. |
| Decoder | 4 blocchi 1×1 → ConvT 3×3 s2 → 1×1 (BN+ReLU), somma con la skip | Come MAResU-Net |
| Head | ConvT 4×4 s2 → conv 3×3 → conv 3×3 (float32) | Logit sempre in float32 anche con mixed precision |

Correzioni rispetto al 2022, garantite dal design:
- nessun modulo costruito come argomento di default (bug V1): ogni `build_*` crea istanze nuove;
- lo stato EMA del codebook sta in variabili non addestrabili aggiornate con `assign` (bug V2);
- la VQ loss viene registrata con `add_loss` e sommata da Keras alla CE, senza clipping euristici (V3).

### 4.3 Self-supervised (`rsseg.ssl`)
`DINOModel` è un `keras.Model` con `train_step` scritto per il backend PyTorch (`loss.backward()`, `optimizer.apply`):

```
viste globali (2×224) ─► teacher (no grad) ─► GAP ─► head ─► softmax((z − c)/τ_t)
tutte le viste       ─► student            ─► GAP ─► head ─► log_softmax(z/τ_s)
loss = media sul batch di Σ_{g} Σ_{v≠g} CE(P_t(g), P_s(v)) / n_coppie        (per campione)
teacher ← m·teacher + (1−m)·student   (m: coseno 0.996 → 1, include le statistiche BN)
c ← λ·c + (1−λ)·media_batch(z_teacher)
```

- Il teacher nasce come copia esatta dello student (`set_weights`) ed è `trainable=False`.
- Schedule coseno per learning rate (con warmup e scaling `lr × batch/256`), weight decay (0.04 → 0.4) e momentum.
  Warmup lineare della temperatura del teacher. `global_clipnorm`.
- L'ultimo layer della head è una `WeightNormDense` (colonne normalizzate, come `norm_last_layer=True`).
- A fine training viene esportato **solo il backbone dello student** in `.weights.h5`, caricabile con `model.weights=<path>`.
- `teacher_entropy` viene loggata a ogni epoca: se tende a 0 il modello sta collassando.

### 4.4 Training supervisionato (`rsseg.train`)
- `model.compile(optimizer=AdamW(gradient_accumulation_steps=...), loss=MaskedSparseCrossEntropy, metrics=[miou, mf1, oa, miou_no_clutter])`.
- La loss ignora **solo** il valore `255`: la classe clutter viene addestrata. C'è un test di regressione per il bug S1.
- Le metriche sono `keras.metrics.Metric` che accumulano una matrice di confusione `float64` sull'intero split.
- Scheduler per epoca: `step` (StepLR 15/0.4 come nella tesi), `cosine`, `constant`.
- Callback, in ordine: `LearningRateScheduler`, `ModelCheckpoint` (miglior `val_miou`, solo pesi), `CodebookMonitor`,
  `GPUMemoryLogger`, `PredictionLogger`, `CSVLogger`, `MLflowMetricsLogger` (ultimo, così registra anche le
  voci aggiunte dagli altri).
- A fine training si ricaricano i pesi migliori, si valuta su tile intere e si registra il report su MLflow.

### 4.5 Valutazione (`rsseg.eval`)
- `predict_tile`: finestre (512, stride 256) con fusione delle softmax pesata da una finestra di Hann e padding
  riflesso per le tile piccole. Il buffer delle probabilità sta sulla CPU, sulla GPU va un batch di finestre alla
  volta. TTA con flip opzionale.
- Il report ha lo stesso formato delle tabelle 4.3–4.8 della tesi (IoU, F1, precision e recall per classe; OA, mF1,
  mIoU), calcolato sia con tutte le classi sia senza clutter. `rsseg report` aggrega media ± deviazione standard sui seed.
- `rsseg benchmark` esegue passi di training reali su dati casuali e misura il picco di VRAM e le immagini al secondo.

## 5. Configurazione

Le YAML ereditano con `base:` e i valori si sovrascrivono da riga di comando:

```bash
rsseg train -c configs/experiments/maresunet_vq_r18_imagenet.yaml \
  --set train.batch_size=4 train.grad_accum_steps=8 model.vq.num_codes=256 name=vq256
```

Le sezioni sono `data`, `model` (con `vq`), `train`, `eval`, `ssl`, `tracking`: vedi `src/rsseg/config.py`, dove ogni
campo è documentato.

## 6. Qualità

| Livello | Cosa si verifica |
|---|---|
| Equivalenza | Backbone Keras = torchvision (R18/R34, pesi casuali, statistiche BN casuali). LinearPAM = trascrizione NumPy della formula originale e = attenzione esplicita `1 + q̂·k̂`. CAM = riferimento NumPy. |
| Invarianti | Split disgiunti, LUT biunivoca, allineamento immagine/maschera dopo l'augmentation, `label_fraction` deterministico |
| VQ | Gradiente straight-through = identità. Loss solo in training. L'EMA ritrova i centri di cluster noti con perplexity alta. |
| DINO | Loss = media delle loss per campione (nessun mescolamento). Teacher = copia iniziale, non addestrabile, aggiornato per EMA. Schedule. |
| Metriche | Metriche Keras accumulate su più batch = NumPy sulla stessa matrice di confusione |
| End-to-end | synth → prepare → train → report → MLflow → pretrain DINO → fine-tuning dal backbone esportato |
| CI | GitHub Actions: ruff (lint + format) e pytest su Python 3.11/3.12 con PyTorch CPU |

## 7. Licenza

Il codice è distribuito sotto **GNU AGPL v3.0 o successiva** (`AGPL-3.0-or-later`, vedi `LICENSE`). Ogni file sorgente
ha l'intestazione SPDX. Il materiale in `legacy/` mantiene la propria licenza (Apache-2.0 per il codice,
tutti i diritti riservati per tesi e slide). Pesi torchvision e dataset ISPRS seguono le rispettive condizioni.
