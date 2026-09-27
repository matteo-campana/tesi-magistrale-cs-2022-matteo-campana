# Piano di sviluppo

Piano per reimplementare **da zero** il lavoro della tesi, secondo l'[architettura](ARCHITECTURE.md): Keras 3 su
PyTorch, tracking con MLflow, eseguibile su una singola RTX 4060 da 8 GB ([GPU_LOCALE.md](GPU_LOCALE.md)).

Legenda: ✅ fatto e verificato (test/CI) · 🟡 implementato, da eseguire su dati reali/GPU · ⬜ da fare.

## Stato sintetico

| Fase | Codice | Esperimenti su dati reali |
|---|---|---|
| F0 Fondamenta | ✅ | — |
| F1 Pipeline dati | ✅ | ✅ Potsdam e Vaihingen preparati, pool Toronto |
| F2 Baseline MAResU-Net | ✅ | 🟡 |
| F3 Vector Quantization | ✅ | 🟡 ablation |
| F4 Pre-training DINO | ✅ | 🟡 |
| F5 Campagne di esperimenti | ✅ `rsseg campaign` | 🟡 prove `--quick` e benchmark su RTX 4060; campagne complete da lanciare |
| F6 Report e rilascio | ⬜ | ⬜ |
| F7 Vaihingen, nDSM, pool DINO | ✅ | 🟡 |

---

## Fase 0: Fondamenta
- ✅ Materiale originale spostato in `legacy/` con `git mv` (storia preservata), `legacy/README.md`
- ✅ `__pycache__/` e `.vs/` rimossi; `.gitignore` per `data/`, `runs/`, `mlflow.db`, pesi
- ✅ `pyproject.toml` (pacchetto `rsseg`, layout `src/`, entry point `rsseg`)
- ✅ `ruff` (lint + format), `pytest`, CI GitHub Actions (Python 3.11/3.12, PyTorch CPU)
- ✅ Intestazioni SPDX `AGPL-3.0-or-later`, `CITATION.cff`
- ⬜ `mypy`, `pre-commit`, `CONTRIBUTING.md`

## Fase 1: Pipeline dati
- ✅ `classes.py`: LUT vettorizzata, `ignore_index=255`, mapping dall'ordine legacy
- ✅ `splits.py`: protocolli `standard` e `thesis2022`
- ✅ `prepare.py`: split per tile → tiling → manifest con istogrammi; statistiche solo sul train
- ✅ `CropDataset`, `MultiCropDataset`, augmentation congiunte, `label_fraction` per tile
- ✅ Generatore di tile sintetiche (`rsseg synth`) per test e demo
- ✅ Split `standard` verificato: le 24 tile di train coincidono con `5_Labels_for_participants`
- ✅ Dati reali in `data/download/`: sorgenti esplicite con priorità delle etichette (`sources.py`), rifiuto delle
  etichette corrotte (Potsdam 4_12 e 6_7 in `5_Labels_all`), pad di DSM/nDSM di 3_13, `rsseg prepare --dry-run`
- ✅ Maschere eroded per val/test e report "full + eroded", con definizioni allineate al tool ufficiale ISPRS
- ⬜ EDA: distribuzione delle classi per tile e split (per documentare perché la tesi aveva rimosso 10 aree)

## Fase 2: Baseline MAResU-Net (RQ1)
- ✅ Backbone ResNet18/34/50 in Keras con pesi ImageNet portati da torchvision (equivalenza testata)
- ✅ LinearPAM + CAM (testati contro la formula della tesi), decoder, varianti `lam`/`plain`
- ✅ CE che addestra anche clutter; metriche su matrice di confusione globale; valutazione su tile intere
- ✅ MLflow: parametri, metriche per epoca, VRAM, predizioni, checkpoint, report
- 🟡 Esperimenti:
  - `thesis2022_maresunet_r18_imagenet`: confronto diretto con la tab. 4.5 (mIoU 0.604)
  - `maresunet_r18_imagenet`, `maresunet_r34_imagenet` sul protocollo standard
  - `unet_r18_imagenet`: senza LAM, per la RQ1

**Fatto quando:** ci sono risultati su ≥ 3 seed, la classe clutter ha IoU > 0 e le differenze rispetto alla tesi sono spiegate.

## Fase 3: Vector Quantization (RQ2)
- ✅ `VectorQuantizer`: EMA in variabili, init dai dati, restart dei codici morti, straight-through, perplexity e usage
- ✅ Neck `vq`, quantizzazione opzionale delle skip (`vq.on_skips`), `CodebookMonitor` su MLflow
- 🟡 Ablation: `num_codes ∈ {64, 256, 512, 1024}`, `beta ∈ {0.1, 0.25, 1.0}`, `loss_weight`, `on_skips`, `ema_decay=0`

  Nei run sintetici il codebook del bottleneck tende a usare pochi codici quando le skip portano già
  l'informazione. È l'ipotesi "il decoder aggira il bottleneck" e va misurata su Potsdam.

## Fase 4: Pre-training DINO (RQ3, RQ4)
- ✅ Head con `WeightNormDense`, loss per campione, centering, schedule (lr/wd/momentum/τ_t), EMA teacher
- ✅ Multi-crop 2×224 + N×96, variante `aerial_augment`
- ✅ `teacher_entropy` (anti-collasso), export del backbone e caricamento in `model.weights`
- 🟡 Esperimenti: fine-tuning con `label_fraction ∈ {0.01, 0.1, 0.25, 1.0}` × {DINO, ImageNet, nessun pre-training}
  e `ssl.weights=imagenet` (pre-training continuato sul dominio)
- ⬜ Linear probe / k-NN sull'encoder congelato come diagnostica aggiuntiva
- ⬜ Variante densa (loss locale per patch) più adatta alla segmentazione

## Fase 5: Griglia della tesi e ablation (RQ1–RQ5)
- ✅ `rsseg campaign` con campagne YAML per ogni RQ (`configs/campaigns/`): prerequisiti automatici, store condiviso
  con ripresa, errori isolati per passo, report per campagna; `scripts/run_thesis_grid.sh` è un wrapper di
  `thesis_grid.yaml`
- ⬜ Esecuzione della griglia; ablation LAM/plain, augmentation, CE pesata, `aerial_augment`
- 🟡 Profilo di efficienza: campagna `rq5_efficiency` provata in `--quick` (R50 con batch 8 non entra in 8 GB;
  aggiunta la variante a batch 4); tempi di test ed esperimenti misurati in [ESECUZIONE_E_TEMPI.md](ESECUZIONE_E_TEMPI.md)
- ✅ Valutazione su tile sotto `torch.no_grad()`: prima il grafo autograd mandava R34 in OOM e rendeva la
  valutazione 3 volte più lenta (ViT-B/16: 10 → 0,6 min per tile)
- ⬜ ViT-B/16 costa ~30 min per epoca su RTX 4060: decidere se ridurre le epoche o escluderlo dalla campagna `transformers`

## Fase 7: Vaihingen, nDSM e pool non etichettati
- ✅ Vaihingen come secondo benchmark: split 16/17 (val 11, 30), config `configs/datasets/vaihingen.yaml`,
  `configs/experiments/vaihingen_*`
- ✅ nDSM come 4° canale (RQ6): nDSM ufficiale di Potsdam o calcolato dal DSM (`rsseg.data.ndsm`), crop sidecar,
  statistiche sul train, estensione del primo layer degli encoder (`mean`/`zeros`)
- ✅ `rsseg prepare-unlabeled` (Toronto, frame di Vaihingen a 16 bit), `ssl.extra_pools` e guardia anti-leakage
  per città
- 🟡 Esperimenti: RGB vs RGB+nDSM su Potsdam e Vaihingen; nDSM ufficiale vs calcolato; DINO con e senza pool
- ⬜ Cross-dataset: Potsdam IRRG → Vaihingen (stesso falso colore)

## Fase 6: Documentazione e rilascio
- ⬜ `docs/RISULTATI.md`: confronto tesi 2022 vs reimplementazione, risposta a ciascuna RQ
- ⬜ Model card e pesi dei modelli migliori (rispettando le condizioni ISPRS)
- ⬜ Tag `v1.0.0`, `CHANGELOG.md`

---

## Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| 8 GB di VRAM | bf16, accumulo del gradiente, crop 512, `rsseg benchmark` prima dei run lunghi, leve documentate in GPU_LOCALE §4; un OOM ferma solo quel passo della campagna |
| Tempi lunghi (tutte le campagne ≈ 13 giorni su RTX 4060) | Early stopping, run condivisi tra campagne, `--seeds 0` prima, ordine consigliato in ESECUZIONE_E_TEMPI §4.2 |
| Channels-last di Keras su backend PyTorch (trasposizioni interne alle conv) | Costo accettabile. Se il profilo lo indica, si può valutare `image_data_format=channels_first`. |
| Lo split standard non coincide con altre fonti | Lo split è un dato di configurazione, verificato e citato. Il protocollo della tesi resta disponibile. |
| DINO collassa con pochi dati | `teacher_entropy`, warmup, partenza da ImageNet |
| Risultati di nuovo negativi | Con protocollo corretto e più seed, anche un risultato negativo diventa una conclusione solida |

## Tracciabilità tesi → codice

| Concetto nella tesi | Sezione | Codice |
|---|---|---|
| Linear Attention Mechanism | §2.5 | `rsseg.models.attention.LinearPAM` |
| MAResU-Net | §2.6 | `rsseg.models.segmentor.build_segmentor` (`skip: lam`) |
| VQ-VAE codebook e loss | §2.4, §4.2.1 | `rsseg.models.quantizer.VectorQuantizer` |
| Self-distillation, teacher EMA | §3 | `rsseg.ssl.dino.DINOModel` |
| DINO loss e multi-crop | §3.2, §4.2.1 | `rsseg.ssl.dino.dino_loss`, `rsseg.data.transforms.MultiCropAugment` |
| Dataset Potsdam, pre-processing | §4.1 | `rsseg.data.prepare`, `rsseg.data.splits` |
| mIoU, pixel accuracy, F1 | §4.2.2 | `rsseg.train.metrics` |
| Tabelle dei risultati | §4.3 | `rsseg.eval.report` |
| Efficienza energetica | cap. 5 | `rsseg.eval.benchmark` |
