# Revisione del codice originale (2022)

Questa è una ricognizione del codice presente nel repository al momento della ripartenza. Serve a due scopi:
capire **come** sono stati prodotti i risultati della tesi ed evitare di **riportare nella nuova
implementazione** i difetti trovati.

> Il codice originale resta nel repository come riferimento storico, con licenza Apache-2.0 (vedi i file
> `LICENSE` nelle rispettive cartelle). Il nuovo codice sarà scritto da zero e non ne copierà parti.

## 1. Inventario

| Cartella | Contenuto | Note |
|---|---|---|
| `tesi-magistrale-main/` | PDF della tesi (V4), slide (PDF/PPTX), testo della presentazione (PDF/DOCX) | Documento di riferimento |
| `semantic-segmentation-potsdam-main/` | Versione "ripulita" per Potsdam: `main_pl_ss.py`, `LitAutoEncoder_SS.py`, modelli MAResUNet18/50 (± VQ), metriche, dataset `.npy` | ~2.900 righe, solo fine-tuning supervisionato |
| `image-segmentation-with-swin-transformers-main/` | Repository di lavoro completo: pre-training DINO (`main_pl_ss_dino.py`, `aux_train/`), modelli Swin/MASWinTNet, VAE/VQ-VAE ResNet, notebook di preparazione dati (Vaihingen), script di lancio | ~14.400 righe, contiene anche esperimenti con Swin Transformer **non riportati nella tesi** |

Stack originale: Python 3.7, PyTorch 1.10 (cu111), PyTorch Lightning 1.5.8, timm 0.4.12, numba, Google Colab.

## 2. Flusso di esecuzione originale

```
config_changer.py (riscrive la YAML) ─► main_pl_ss.py
                                          ├─ legge config/train_pl_ss.yaml
                                          ├─ LitAutoEncoder(build_model(name))
                                          ├─ trainer.tune()  → auto_scale_batch_size="binsearch"
                                          └─ trainer.fit()   → TensorBoard (loss, mIoU, OA, dice, immagini)

main_pl_ss_dino.py
  ├─ fase 1: LitFeaturesExtractor (teacher/student = 2× build_model, output a 3 canali)
  │          loss DINOLoss su output denso, teacher EMA α = 0.996
  └─ fase 2: sostituisce finalconv3 → copia lo student in LitAutoEncoder → fine-tuning
```

## 3. Difetti trovati

Gravità: 🔴 invalida o distorce i risultati · 🟠 si discosta da quanto dichiarato nella tesi · 🟡 qualità/manutenibilità.

### 3.1 Training supervisionato
| # | Gravità | Dove | Problema | Effetto |
|---|---|---|---|---|
| S1 | 🔴 | `LitAutoEncoder_SS.py` (`nn.CrossEntropyLoss(ignore_index=0)`) | La classe 0 è **Clutter/Background** (vedi `utils/dataset_utils.py`) e viene esclusa dalla loss. | Il modello non impara mai la classe Clutter. È coerente con i punteggi molto bassi di quella classe in tutte le tabelle. |
| S2 | 🔴 | `metrics.py`, `*_epoch_end` | Le metriche sono calcolate **per batch** e poi mediate, con `nanmean` sulle classi assenti. | Valori gonfiati e non confrontabili. Serve una matrice di confusione accumulata su tutto lo split. |
| S3 | 🟠 | `train_dataloader` | Nessuno `shuffle=True`. | Batch ordinati per file, quindi spazialmente correlati: ottimizzazione peggiore. |
| S4 | 🟠 | `DatasetPotsdamSemantiSegmentatin` | Input solo `/255`, **nessuna normalizzazione** con media e deviazione standard ImageNet. | Mismatch con i pesi ImageNet del backbone. La tesi dichiara una normalizzazione. |
| S5 | 🟠 | `DatasetPotsdamSemantiSegmentatin` | `T.Resize` applicato **anche alla maschera** con interpolazione bilineare di default. | Etichette interpolate (valori non interi) ai bordi delle classi. |
| S6 | 🟡 | `val_dataloader` (Potsdam) | Usa `self.train_dataset_length` invece di `valid_dataset_length`. | Innocuo con -1, sbagliato altrimenti. |
| S7 | 🟡 | `training_step` | `F.mse_loss(output, target)` tra logit a 6 canali e maschera a 1 canale (broadcast). | Metrica "mse" priva di significato. |
| S8 | 🟡 | `main_pl_ss.py` | Nessun seed, nessun checkpoint "best", nessuno step di test. | Risultati non riproducibili, e non si sa da dove vengano i "Test Scores". |

### 3.2 Vector Quantization
| # | Gravità | Dove | Problema | Effetto |
|---|---|---|---|---|
| V1 | 🔴 | `models/MAResUNet_vq.py`: `def __init__(..., base_model=tmodels.resnet18(pretrained=True), ...)` | Argomento di default **valutato una sola volta all'import**: tutte le istanze `MAResUNet18_VQ_SS` condividono lo **stesso** ResNet. | In DINO-VQ teacher e student condividono l'encoder. Il loop che congela il teacher (`requires_grad=False`) congela anche l'encoder dello student, e l'EMA agisce sugli stessi tensori. |
| V2 | 🟠 | `VectorQuantizerEMA.forward` | `self._ema_w = nn.Parameter(...)` e `self._embedding.weight = nn.Parameter(...)` vengono riassegnati a ogni step. | I parametri registrati nell'ottimizzatore non sono più quelli usati. Serve implementarli come buffer. |
| V3 | 🟠 | `LitAutoEncoder_SS.py` | Loss = CE + VQ loss "clippata" a ±10·CE (`reluctance_to_change_threshold`). | Euristica non documentata. La tesi parla di "MSE + VQ_LOSS". |
| V4 | 🟠 | `MAResUNet18_VQ_SS` | Il codebook (512×512) quantizza **solo** `attention4(e4)`, mentre le skip connection `a1..a3` lo aggirano. | Il decoder può ignorare il bottleneck quantizzato: il test dell'ipotesi è debole. |
| V5 | 🟡 | — | Non si monitorano l'uso del codebook né i codici morti (la perplexity viene loggata ma non analizzata). | Impossibile capire se il codebook collassa. |

### 3.3 Pre-training DINO
| # | Gravità | Dove | Problema rispetto a DINO originale |
|---|---|---|---|
| D1 | 🔴 | `aux_train/dino_loss.py` | La softmax è calcolata su `teacher_output.view(1, -1)`, cioè su **tutto il batch appiattito** (B×2×3×224×224 valori). Distribuzioni di campioni diversi si mescolano, e anche il centro ha quella dimensione. |
| D2 | 🔴 | `LitFeaturesExtractor` | Non c'è **projection head** né output a K prototipi: la loss agisce sull'output **denso a 3 canali del decoder**. Si pre-addestra tutto l'encoder-decoder, non l'encoder. |
| D3 | 🟠 | `LitFeaturesExtractor` | Teacher e student sono costruiti con due chiamate indipendenti a `build_model`, quindi il teacher non è una copia dello student (decoder diversi). |
| D4 | 🟠 | `dino_augment_transforms.py` | I crop locali vengono ridimensionati a **224** (la riga a 96 è commentata). La normalizzazione è ImageNet, non quella dichiarata nella tesi (§4.1.2). |
| D5 | 🟠 | `LitFeaturesExtractor` | Temperatura del teacher fissa a 0.04 (warmup 0; la tesi dice 0.04→0.07). Adam senza weight decay (la tesi dice cosine 0.04→0.4). Momentum EMA costante a 0.996 (DINO usa un coseno 0.996→1). Nessuno scheduler del learning rate. |
| D6 | 🟡 | `DINOLoss` | Il doppio loop Python su batch e crop è molto lento. Non si esclude il confronto tra una vista e sé stessa (nel codice le viste locali e globali sono comunque disgiunte). |

### 3.4 Dati
| # | Gravità | Dove | Problema |
|---|---|---|---|
| P1 | 🔴 | Split | Nel repository non c'è uno script di preparazione per Potsdam (il notebook `creazione_dataset.ipynb` è per Vaihingen e usa stride = 128, cioè crop **sovrapposti al 50%**). Se lo stesso schema è stato usato per Potsdam con split non disgiunti per area, la validation si sovrappone al training. È compatibile con mIoU di validation ≈ 0.96. |
| P2 | 🟠 | Tesi §4.1.2 | Sono rimosse 10 aree, tra cui 2_13, 2_14, 6_13 e 7_13, che nello split di test più usato in letteratura sono tile di test. Il confronto con lo stato dell'arte non è possibile. |
| P3 | 🟡 | `to_class_channels_jit` e simili | Conversione RGB→classe con loop numba pixel per pixel. Si può vettorizzare con una LUT. |

### 3.5 Ingegneria
- `config_changer.py` sovrascrive la YAML in-place e attende un `input()` (da qui `yes |` negli script).
- Molto codice duplicato tra le due cartelle e tra le varianti (VQ e non VQ copiate e incollate).
- Percorsi Colab/Google Drive cablati nel codice, file `__pycache__` e `.vs/` versionati.
- Nessun test automatico, nessun lint, nessuna CI.
- Dipendenze bloccate a versioni del 2021 (PL 1.5 usa API rimosse: `training_epoch_end`, `gpus=`, `trainer.tune`).

## 4. Cosa riusare concettualmente (non il codice)

- La struttura di MAResU-Net: encoder ResNet, `PAM_CAM_Layer` (PAM lineare + CAM) sulle 4 scale,
  `DecoderBlock` 1×1 → deconv 3×3 s2 → 1×1, testa finale deconv 4×4 s2 → conv 3×3 → conv 3×3.
- La formula del PAM lineare (normalizzazione L2 di Q e K, termine `N + qᵢᵀ Σ kⱼ`, `gamma` inizializzato a 0).
- Mappatura colori/classi e metriche (OA, F1/Dice, IoU per classe).
- L'idea del flusso in due fasi (pre-training self-supervised, poi fine-tuning).
