# Semantic Segmentation for Remote Sensing Images

Reimplementazione da zero della tesi di laurea magistrale
**"Semantic Segmentation for Remote Sensing Images"**: Matteo Campana, Engineering in Computer Science,
Sapienza Università di Roma, A.A. 2021/2022. Relatore Prof. Aurelio Uncini, correlatori
Dr. Simone Scardapane e Dr. Valerio Marsocci.

> **Stato del progetto:** fase di analisi e progettazione completata. Il codice `rsseg` è in sviluppo secondo il
> [piano](docs/PIANO_DI_SVILUPPO.md). I comandi riportati sotto descrivono l'interfaccia prevista.

## Di cosa si tratta

Segmentazione semantica di ortofoto aeree ad altissima risoluzione (ISPRS **Potsdam**, 5 cm/pixel, 6 classi:
superfici impermeabili, edifici, vegetazione bassa, alberi, auto, clutter).

La tesi parte dall'architettura **MAResU-Net** (ResNet + U-Net + *Linear Attention Mechanism* a costo O(N)),
usata dal lavoro [MARE](https://www.mdpi.com/2072-4292/13/16/3275) dei correlatori, e studia due estensioni:

| Idea | Motivazione |
|---|---|
| **Bottleneck quantizzato (VQ-VAE)** | Rappresentazioni discrete e compatte, sul modello dei *visual words* di OBoW usato in MARE |
| **Pre-training self-supervised DINO** | Etichette costose e scarse nel telerilevamento: sfruttare le immagini non etichettate |

Nella tesi originale nessuna delle due estensioni ha battuto la baseline con pesi ImageNet
(miglior test mIoU ≈ 0.604 con MAResU-Net-R18). L'[analisi](docs/ANALISI_TESI.md) mostra però che il
protocollo e l'implementazione del 2022 avevano problemi che rendono quel risultato **non conclusivo**: la classe
Clutter era esclusa dalla loss, la validation probabilmente non era indipendente e DINO non era implementato
come nell'originale. Questo repository ripropone le stesse domande di ricerca con un'implementazione corretta,
riproducibile e confrontabile con la letteratura.

## Documentazione

| Documento | Contenuto |
|---|---|
| [docs/ANALISI_TESI.md](docs/ANALISI_TESI.md) | Contenuto della tesi, motivazioni delle scelte, risultati, lettura critica, domande di ricerca |
| [docs/REVISIONE_CODICE_ORIGINALE.md](docs/REVISIONE_CODICE_ORIGINALE.md) | Ricognizione del codice 2022 e difetti trovati |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architettura del nuovo pacchetto `rsseg` |
| [docs/DATASET.md](docs/DATASET.md) | ISPRS Potsdam, classi, split, pre-processing, augmentation |
| [docs/PIANO_DI_SVILUPPO.md](docs/PIANO_DI_SVILUPPO.md) | Fasi, deliverable, criteri di completamento, rischi |

## Contenuto attuale del repository

| Percorso | Descrizione |
|---|---|
| `tesi-magistrale-main/` | Tesi (PDF), slide e testo della presentazione: il **documento di riferimento** |
| `semantic-segmentation-potsdam-main/` | Codice originale 2022: fine-tuning MAResU-Net su Potsdam (Apache-2.0) |
| `image-segmentation-with-swin-transformers-main/` | Codice originale 2022: DINO, VQ, esperimenti Swin (Apache-2.0) |
| `docs/` | Documentazione del progetto |

Nella fase 0 il materiale originale sarà spostato in `legacy/` e il nuovo codice andrà in `src/rsseg/`.

## Uso previsto

```bash
# ambiente
uv sync

# 1. dati (il dataset ISPRS va scaricato separatamente, vedi docs/DATASET.md)
rsseg prepare --config configs/data/potsdam_standard.yaml --raw_dir /path/to/Potsdam

# 2. (opzionale) pre-training self-supervised dell'encoder
rsseg pretrain --config configs/ssl/dino_r18.yaml

# 3. training e valutazione
rsseg fit  --config configs/experiments/maresunet_r18_imagenet.yaml --seed_everything 0
rsseg test --config configs/experiments/maresunet_r18_imagenet.yaml --ckpt_path best

# 4. tabelle riassuntive
rsseg report --runs runs/ --out reports/summary.md
```

## Risultati della tesi originale (test, Potsdam, protocollo 2022)

| Modello | Backbone | Pre-train | OA | mF1 | mIoU |
|---|---|---|---|---|---|
| MAResU-Net | ResNet18 | ImageNet | 0.791 | 0.722 | **0.604** |
| MAResU-Net | ResNet34 | ImageNet | 0.791 | 0.722 | 0.604 |
| MAResU-Net | ResNet18 | DINO | 0.741 | 0.657 | 0.530 |
| MAResU-Net-VQ | ResNet18 | ImageNet | 0.750 | 0.668 | 0.543 |
| MAResU-Net-VQ | ResNet18 | DINO | 0.702 | 0.593 | 0.465 |

Tabella completa e analisi in [docs/ANALISI_TESI.md](docs/ANALISI_TESI.md#53-risultati-di-test-tab-45).
I risultati della reimplementazione saranno pubblicati in `docs/RISULTATI.md`.

## Dati

Il dataset ISPRS Potsdam **non** è incluso e non va committato. Va richiesto e scaricato dal sito ISPRS
rispettandone le condizioni d'uso. Vedi [docs/DATASET.md](docs/DATASET.md).

## Citazione

Se usi questo lavoro, cita la tesi (vedi [`CITATION.cff`](CITATION.cff)) e i lavori su cui si basa:

- V. Marsocci, S. Scardapane, N. Komodakis. *MARE: Self-Supervised Multi-Attention REsu-Net for Semantic
  Segmentation in Remote Sensing.* Remote Sensing, 2021.
- R. Li et al. *Multistage Attention ResU-Net for Semantic Segmentation of Fine-Resolution Remote Sensing
  Images.* IEEE GRSL, 2022.
- A. van den Oord et al. *Neural Discrete Representation Learning.* NeurIPS, 2017.
- M. Caron et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV, 2021.

## Licenza

Il nuovo codice e la documentazione di questo repository sono distribuiti sotto
**GNU Affero General Public License v3.0 o successiva** (`AGPL-3.0-or-later`). Vedi [`LICENSE`](LICENSE).

In sintesi: puoi usare, modificare e ridistribuire il software, ma le versioni modificate, **anche quando sono
solo offerte come servizio in rete**, devono rendere disponibile il codice sorgente sotto la stessa licenza.

Eccezioni:
- il codice originale del 2022 nelle cartelle `*-main/` (future `legacy/`) resta sotto la propria licenza **Apache-2.0**;
- il testo della tesi e le slide sono © 2022 Matteo Campana, tutti i diritti riservati, come indicato nel documento;
- dataset ISPRS e pesi pre-addestrati di terzi seguono le rispettive licenze.

## Ringraziamenti

Prof. Aurelio Uncini, Dr. Simone Scardapane e Dr. Valerio Marsocci per la guida durante la tesi.
ISPRS per il dataset Potsdam.
