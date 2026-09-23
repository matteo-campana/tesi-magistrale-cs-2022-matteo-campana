# Piano di sviluppo

Piano per reimplementare **da zero** il lavoro della tesi, secondo l'[architettura](ARCHITECTURE.md): Keras 3 su
PyTorch, tracking con MLflow, eseguibile su una singola RTX 4060 da 8 GB ([GPU_LOCALE.md](GPU_LOCALE.md)).

Legenda: ✅ fatto e verificato (test/CI) · 🟡 implementato, da eseguire su dati reali/GPU · ⬜ da fare.

## Stato sintetico

| Fase | Codice | Esperimenti su Potsdam |
|---|---|---|
| F0 Fondamenta | ✅ | — |
| F1 Pipeline dati | ✅ | 🟡 `rsseg prepare` sul dataset reale |
| F2 Baseline MAResU-Net | ✅ | 🟡 |
| F3 Vector Quantization | ✅ | 🟡 ablation |
| F4 Pre-training DINO | ✅ | 🟡 |
| F5 Griglia + ablation | ✅ script | ⬜ |
| F6 Report e rilascio | ⬜ | ⬜ |

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
- 🟡 Verificare lo split `standard` sulle fonti ufficiali ISPRS/MMSegmentation
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
- ✅ `scripts/run_thesis_grid.sh` (2 × 2 × 2 × seed) e `rsseg report` (media ± std)
- ⬜ Esecuzione della griglia; ablation LAM/plain, augmentation, CE pesata, `aerial_augment`
- ⬜ Profilo di efficienza con `rsseg benchmark` (parametri, immagini/s, VRAM) per R18/R34/R50

## Fase 6: Documentazione e rilascio
- ⬜ `docs/RISULTATI.md`: confronto tesi 2022 vs reimplementazione, risposta a ciascuna RQ
- ⬜ Model card e pesi dei modelli migliori (rispettando le condizioni ISPRS)
- ⬜ Tag `v1.0.0`, `CHANGELOG.md`

---

## Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| 8 GB di VRAM | bf16, accumulo del gradiente, crop 512, `rsseg benchmark` prima dei run lunghi, leve documentate in GPU_LOCALE §4 |
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
