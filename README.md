# Semantic Segmentation for Remote Sensing Images

Reimplementazione da zero della tesi di laurea magistrale
**"Semantic Segmentation for Remote Sensing Images"**: Matteo Campana, Engineering in Computer Science,
Sapienza Università di Roma, A.A. 2021/2022. Relatore Prof. Aurelio Uncini, correlatori
Dr. Simone Scardapane e Dr. Valerio Marsocci.

**Keras 3 su backend PyTorch · tracking con MLflow · eseguibile su una singola GPU NVIDIA RTX 4060 (8 GB).**

> **Stato:** il codice è completo per tutte le parti della tesi (MAResU-Net, VQ, DINO, valutazione) ed è verificato
> da 32 test automatici su CPU e da run end-to-end su dati sintetici. Gli esperimenti sul dataset Potsdam reale sono
> il prossimo passo: vedi il [piano](docs/PIANO_DI_SVILUPPO.md).

## Di cosa si tratta

Segmentazione semantica di ortofoto aeree ad altissima risoluzione (ISPRS **Potsdam**, 5 cm/pixel, 6 classi:
superfici impermeabili, edifici, vegetazione bassa, alberi, auto, clutter).

La tesi parte dall'architettura **MAResU-Net** (ResNet + U-Net + *Linear Attention Mechanism* a costo O(N)),
usata dal lavoro [MARE](https://www.mdpi.com/2072-4292/13/16/3275) dei correlatori, e studia due estensioni:

| Idea | Motivazione |
|---|---|
| **Bottleneck quantizzato (VQ-VAE)** | Rappresentazioni discrete e compatte, sul modello dei *visual words* di OBoW usato in MARE |
| **Pre-training self-supervised DINO** | Etichette costose e scarse nel telerilevamento: sfruttare le immagini non etichettate |

Nella tesi originale nessuna delle due ha battuto la baseline ImageNet (miglior test mIoU ≈ 0.604). L'[analisi](docs/ANALISI_TESI.md)
mostra però che quel risultato **non è conclusivo**: la classe clutter era esclusa dalla loss, la validation probabilmente
non era indipendente e DINO non era implementato come nell'originale. Questo repository ripropone le stesse domande con
un'implementazione corretta, riproducibile e confrontabile con la letteratura.

## Avvio rapido

```bash
# PyTorch con CUDA (scegli l'indice per il tuo driver su pytorch.org), poi il pacchetto
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"

# prova completa senza dataset, su tile sintetiche (~2 minuti)
rsseg synth   --out data/synthetic_raw --size 256
rsseg prepare -c configs/smoke.yaml
rsseg train   -c configs/smoke.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

Con il dataset reale (va scaricato da ISPRS, vedi [docs/DATASET.md](docs/DATASET.md)):

```bash
rsseg prepare   -c configs/base.yaml --set data.raw_dir=/percorso/Potsdam
rsseg benchmark -c configs/experiments/maresunet_r18_imagenet.yaml   # entra in 8 GB? misura il picco di VRAM
rsseg train     -c configs/experiments/maresunet_r18_imagenet.yaml   # training + valutazione su tile di test
rsseg pretrain  -c configs/ssl/dino_r18.yaml                         # DINO sull'encoder
rsseg train     -c configs/experiments/maresunet_r18_dino.yaml       # fine-tuning dal backbone DINO
bash scripts/run_thesis_grid.sh                                      # griglia 2×2×2 della tesi × 3 seed
```

Guida completa per la RTX 4060 (installazione, memoria, MLflow, problemi comuni): [docs/GPU_LOCALE.md](docs/GPU_LOCALE.md).

## Comandi

| Comando | Cosa fa |
|---|---|
| `rsseg synth` | Genera tile sintetiche con gli stessi nomi di file di Potsdam |
| `rsseg prepare` | Split per tile, conversione delle etichette, crop, manifest, statistiche |
| `rsseg train` | Training con Keras `fit` + valutazione su tile intere + log su MLflow |
| `rsseg pretrain` | Pre-training DINO dell'encoder ed export del backbone |
| `rsseg evaluate` | Valuta un file di pesi su tile intere (finestra scorrevole) |
| `rsseg benchmark` | Picco di VRAM e immagini/s di una configurazione |
| `rsseg report` | Tabella media ± std su più run (es. seed) |

Ogni valore di configurazione si può sovrascrivere: `--set train.batch_size=4 train.grad_accum_steps=8 model.arch=resnet34`.

## Cosa c'è dentro

| Parte | Implementazione |
|---|---|
| Backbone | ResNet18/34/50 in Keras con pesi ImageNet portati da torchvision (equivalenza numerica testata) |
| MAResU-Net | Linear PAM (LAM, O(N)) + channel attention sulle skip, decoder della tesi. Variante `plain` (U-Net) per l'ablation. |
| VQ | Codebook EMA (inizializzato dai dati, restart dei codici morti), straight-through, perplexity/usage su MLflow, quantizzazione opzionale delle skip |
| DINO | Head con prototipi, loss per campione, teacher EMA, schedule coseno, multi-crop 2×224 + N×96, monitor di collasso |
| Training | CE che ignora solo i pixel non etichettati, mIoU/mF1/OA da matrice di confusione globale, bf16, accumulo del gradiente |
| Valutazione | Finestra scorrevole su tile 6000² intere, report per classe con e senza clutter, mappe di errore |
| MLflow | Config, metriche per epoca, VRAM, predizioni, checkpoint, report, versioni e commit git |

## Documentazione

| Documento | Contenuto |
|---|---|
| [docs/ANALISI_TESI.md](docs/ANALISI_TESI.md) | Contenuto della tesi, motivazioni delle scelte, risultati, lettura critica, domande di ricerca |
| [docs/REVISIONE_CODICE_ORIGINALE.md](docs/REVISIONE_CODICE_ORIGINALE.md) | Ricognizione del codice 2022 e difetti trovati |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architettura del pacchetto `rsseg` |
| [docs/DATASET.md](docs/DATASET.md) | ISPRS Potsdam, classi, split, pre-processing, augmentation |
| [docs/GPU_LOCALE.md](docs/GPU_LOCALE.md) | Esecuzione su RTX 4060 8 GB e uso di MLflow |
| [docs/PIANO_DI_SVILUPPO.md](docs/PIANO_DI_SVILUPPO.md) | Stato delle fasi, esperimenti da eseguire, rischi |

## Struttura

```
src/rsseg/      pacchetto (data, models, ssl, train, eval, cli)
configs/        base.yaml, smoke.yaml, experiments/, ssl/
scripts/        griglia di esperimenti della tesi
tests/          test automatici (pytest)
docs/           documentazione
legacy/         materiale originale 2022 (tesi, slide, codice) — invariato
```

## Risultati della tesi originale (test, Potsdam, protocollo 2022)

| Modello | Backbone | Pre-train | OA | mF1 | mIoU |
|---|---|---|---|---|---|
| MAResU-Net | ResNet18 | ImageNet | 0.791 | 0.722 | **0.604** |
| MAResU-Net | ResNet34 | ImageNet | 0.791 | 0.722 | 0.604 |
| MAResU-Net | ResNet18 | DINO | 0.741 | 0.657 | 0.530 |
| MAResU-Net-VQ | ResNet18 | ImageNet | 0.750 | 0.668 | 0.543 |
| MAResU-Net-VQ | ResNet18 | DINO | 0.702 | 0.593 | 0.465 |

Tabella completa in [docs/ANALISI_TESI.md](docs/ANALISI_TESI.md#53-risultati-di-test-tab-45). I risultati della
reimplementazione verranno pubblicati in `docs/RISULTATI.md`.

## Sviluppo

```bash
pip install -e ".[dev]"
pytest -q                       # 32 test, ~30 s su CPU
ruff check src tests && ruff format --check src tests
```

## Citazione

Se usi questo lavoro, cita la tesi (vedi [`CITATION.cff`](CITATION.cff)) e i lavori su cui si basa:

- V. Marsocci, S. Scardapane, N. Komodakis. *MARE: Self-Supervised Multi-Attention REsu-Net for Semantic
  Segmentation in Remote Sensing.* Remote Sensing, 2021.
- R. Li et al. *Multistage Attention ResU-Net for Semantic Segmentation of Fine-Resolution Remote Sensing
  Images.* IEEE GRSL, 2022.
- A. van den Oord et al. *Neural Discrete Representation Learning.* NeurIPS, 2017.
- M. Caron et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV, 2021.

## Licenza

Il codice e la documentazione di questo repository sono distribuiti sotto
**GNU Affero General Public License v3.0 o successiva** (`AGPL-3.0-or-later`). Vedi [`LICENSE`](LICENSE).

In sintesi: puoi usare, modificare e ridistribuire il software, ma le versioni modificate, **anche quando sono
solo offerte come servizio in rete**, devono rendere disponibile il codice sorgente sotto la stessa licenza.

Eccezioni:
- il codice originale del 2022 in `legacy/` resta sotto la propria licenza **Apache-2.0**;
- il testo della tesi e le slide in `legacy/tesi-magistrale/` sono © 2022 Matteo Campana, tutti i diritti riservati;
- dataset ISPRS e pesi pre-addestrati di terzi (torchvision) seguono le rispettive licenze.

## Ringraziamenti

Prof. Aurelio Uncini, Dr. Simone Scardapane e Dr. Valerio Marsocci per la guida durante la tesi.
ISPRS per il dataset Potsdam.
