# Dataset: ISPRS Potsdam e Vaihingen

Il progetto usa i dati dell'ISPRS 2D Semantic Labeling Contest: **Potsdam** è il dataset principale (quello della
tesi), **Vaihingen** è il secondo benchmark. Le immagini grezze di Vaihingen e **Toronto** servono solo come dati
non etichettati per il pre-training DINO.

## 1. Descrizione

| Proprietà | Potsdam | Vaihingen |
|---|---|---|
| Tile | 38 true orthophoto (TOP) 6000 × 6000 px | 33 aree di dimensione variabile (~1900 × 2500 px) |
| GSD | 5 cm | 9 cm |
| Immagine usata | **RGB** (`2_Ortho_RGB`) | **IR-R-G** (falso colore, `top/`) |
| Altezze | DSM assoluto + nDSM ufficiale | solo DSM assoluto (l'nDSM viene calcolato, §6) |
| Etichette | 6 classi, piene ed "eroded" | 6 classi, piene ed "eroded" |

Le classi e la palette sono le stesse per i due dataset. Sono mantenute identiche alla tesi per confrontabilità:

| id | Classe | RGB |
|---|---|---|
| 0 | Impervious surfaces | (255, 255, 255) |
| 1 | Building | (0, 0, 255) |
| 2 | Low vegetation | (0, 255, 255) |
| 3 | Tree | (0, 255, 0) |
| 4 | Car | (255, 255, 0) |
| 5 | Clutter/background | (255, 0, 0) |

> ⚠️ **Nota sugli indici.** Il codice originale usa l'ordine `Clutter=0, ImSurf=1, Car=2, Building=3,
> LowVeg=4, Tree=5`. La nuova implementazione adotta l'ordine ISPRS/MMSegmentation qui sopra
> (Clutter per ultimo), così un eventuale `ignore_index` non può colpire per errore una classe reale.
> Il mapping con il vecchio ordine è in `src/rsseg/data/classes.py` (`LEGACY_TO_INDEX`, `legacy_to_index`).

In Vaihingen il clutter è quasi assente: in molte aree è lo 0%. Per Vaihingen la metrica principale è quindi
la mIoU **senza clutter** (`without_clutter` nel report), come in letteratura.

## 2. Licenza e reperimento

I dati **non sono redistribuiti** in questo repository. Vanno scaricati dal sito ISPRS
(<https://www.isprs.org/education/benchmarks/UrbanSemLab/>) accettando le condizioni d'uso. I dati
e le maschere derivate (crop, maschere di tile) non vanno committati: la cartella `data/` è in `.gitignore`.

Gli archivi vanno decompressi in `data/download/`, il `raw_dir` predefinito delle configurazioni:

```
data/download/
├── Potsdam/     1_DSM, 1_DSM_normalisation, 2_Ortho_RGB, 3_Ortho_IRRG, 4_Ortho_RGBIR,
│                5_Labels_all[_noBoundary], 5_Labels_for_participants[_no_Boundary],
│                assess_classification_reference_implementation.tar
├── Vaihingen/   ISPRS_semantic_labeling_Vaihingen/{top,dsm,gts_for_participants},
│                ..._ground_truth_COMPLETE, ..._ground_truth_eroded_COMPLETE, Images, DSM, Ortho, ALS, ...
└── Toronto/     Images, DSM, ALS, Reference_*
```

La struttura interna non conta: le cartelle sono cercate ricorsivamente. Ogni file è riconosciuto da
`src/rsseg/data/sources.py` con espressioni regolari sul percorso relativo, in ordine di priorità. Gli ID
delle tile vengono normalizzati: il DSM di Potsdam usa `02_10`, le immagini `2_10`.

### 2.1 Cosa si usa e cosa no

| Dati | Uso |
|---|---|
| Potsdam `2_Ortho_RGB`, `5_Labels_*`, `1_DSM_normalisation`, `1_DSM` | segmentazione, nDSM |
| Potsdam `3_Ortho_IRRG` | opzionale: `--set "data.image_pattern=top_potsdam_(\d+_\d+)_IRRG\.tif$"` |
| Potsdam `4_Ortho_RGBIR` | non usato (TIFF a 4 bande PackBits, che Pillow non legge) |
| Vaihingen `top`, `dsm`, ground truth `COMPLETE` ed `eroded_COMPLETE` | segmentazione, nDSM calcolato |
| Vaihingen `Images`, Toronto `Images` | pool non etichettati per DINO (§8) |
| ALS (nuvole di punti), `3DLabeling`, `Reference_*` (DXF/SHP), mosaici `Ortho`/`DSM` interi | fuori scope |
| `assess_classification_reference_implementation.tar` | riferimento per le metriche (§7) |

## 3. Problemi noti dei file e come vengono gestiti

L'ispezione dei file scaricati ha trovato:

| Problema | Effetto se ignorato | Gestione |
|---|---|---|
| `5_Labels_all/top_potsdam_4_12_label.tif` (e la versione `_noBoundary`) è corrotto: nessun pixel ha un colore ISPRS | 4_12, una tile di validation, diventa tutta "ignore" | si usa la copia in `5_Labels_for_participants`, che è corretta |
| `5_Labels_all/top_potsdam_6_7_label.tif`: lo 0,7% dei pixel "car" è `(252,255,0)` | quei pixel vengono ignorati | stessa priorità: si usa la copia in `for_participants` |
| DSM e nDSM di Potsdam 3_13 sono larghi 5999 px | disallineamento con l'immagine | pad di 1 colonna (replica del bordo), con warning |
| `Vaihingen/dsm_09cm_matching_area11.tif` fuori dalla cartella `dsm/` | duplicato | ignorato: il pattern richiede la cartella `dsm/` |
| `…eroded_COMPLETE` contiene anche `area9` | nessuno | ignorata: area9 non ha né immagine né split |

Controlli automatici in `rsseg prepare`:

- **priorità delle etichette**: `5_Labels_for_participants` prima di `5_Labels_all`, e lo stesso per le eroded;
- **colori fuori palette**: se più di `data.max_offpalette` (default 0,5%) dei pixel di un'etichetta *piena* non è
  un colore ISPRS, la tile viene rifiutata con un errore che la nomina (le eroded hanno bordi neri per costruzione:
  non vengono controllate);
- **dimensioni**: etichette, nDSM e DSM devono avere la dimensione dell'immagine, con una tolleranza di 2 px.

`rsseg prepare -c CONFIG --dry-run` stampa per ogni tile lo split, la cartella da cui arrivano etichette ed nDSM,
la frazione di colori fuori palette e le anomalie, senza scrivere nulla. Se trova errori esce con codice 1.

## 4. Protocolli di split

Gli split sono sempre per **tile intere** (`src/rsseg/data/splits.py`): nessun crop può finire in due split.

### 4.1 Potsdam `standard` (predefinito, confrontabile con la letteratura)
Split usato comunemente (ad es. da MMSegmentation):

- **train (24 tile)**: 2_10, 2_11, 2_12, 3_10, 3_11, 3_12, 4_10, 4_11, 4_12, 5_10, 5_11, 5_12,
  6_7, 6_8, 6_9, 6_10, 6_11, 6_12, 7_7, 7_8, 7_9, 7_10, 7_11, 7_12. Coincide con le 24 tile di
  `5_Labels_for_participants`.
- **test (14 tile)**: 2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13
- **validation**: un sottoinsieme di tile di training **tenuto fuori per intero** (ad es. 2 tile), mai crop
  della stessa tile.
  Predefinite: 4_12 e 7_11 (`data.val_tiles` per cambiarle).
- Alcuni lavori escludono la tile 7_10 per un errore noto nelle etichette: opzione `data.exclude_7_10: true`.

### 4.2 Potsdam `thesis2022` (riproduzione del protocollo della tesi)
- Rimosse: 2_10, 2_11, 2_12, 2_13, 2_14, 3_10, 6_7, 6_13, 7_13, 7_9.
- Le 28 aree restanti sono divise in 22 train e 6 validation. La tesi non elenca quali: qui la validation è
  fissata a 3_14, 4_15, 5_13, 6_15, 7_11, 4_11 (`THESIS2022_VAL_DEFAULT`). Il "test" coincide con la validation,
  valutata su tile intere.
- Configurazione pronta: `configs/experiments/thesis2022_maresunet_r18_imagenet.yaml` (crop 256, nessuna augmentation, Adam).
- Serve solo a confrontarsi con i numeri della tesi, non per le conclusioni.

### 4.3 Vaihingen `standard` (`data.dataset: vaihingen`)
- **train (16 aree)**: 1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37. Sono le aree date con le
  etichette ai partecipanti al contest (`gts_for_participants`).
- **test (17 aree)**: 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38.
- **validation**: 11 e 30, tenute fuori per intero. La 26 non viene usata perché contiene gran parte del poco
  clutter di training.
- Configurazioni: `configs/datasets/vaihingen.yaml` (dati), `configs/experiments/vaihingen_*.yaml`. Con sole 14
  aree di training, i crop di training si sovrappongono (`train_stride: 256`), solo all'interno del train.

## 5. Pre-processing (`rsseg prepare`)

1. Le tile vengono assegnate agli split **prima** del taglio, così nessun crop attraversa due split.
2. Conversione della maschera RGB in indice di classe con una **LUT vettorizzata**. I colori non in palette diventano
   `255` (ignorato), con il controllo di qualità del §3.
3. Crop:
   - **train**: `crop_size` (512 di default) con `train_stride`. La sovrapposizione è ammessa **solo qui**.
   - **val**: crop non sovrapposti, usati per la validation durante `fit` (selezione del miglior checkpoint).
   - **val/test**: anche la **tile intera** viene registrata in `tiles.csv`, con la maschera piena
     (`tiles/<tile>_mask.png`) e quella eroded (`tiles/<tile>_mask_eroded.png`), per la valutazione finale
     a finestra scorrevole con fusione delle probabilità.
   - I crop interamente non etichettati vengono scartati. L'ultimo crop di ogni riga/colonna è allineato al bordo.
4. Etichette dei crop: `data.train_labels: full` (default, come la tesi) oppure `eroded` (bordi ignorati anche in
   training, come MMSegmentation). È un'opzione di `prepare`: per cambiarla serve un altro `data.root`.
5. Formato: PNG per immagini e maschere, `crops.csv` con istogramma delle classi per crop, `split.json` (split
   e impostazioni: dataset, protocollo, etichette, `crop_size`, `train_stride`, nDSM; `rsseg campaign` lo usa per
   non riusare un root preparato con impostazioni diverse).
6. `stats.json`: media e deviazione standard per canale **solo sul train** (con l'nDSM anche `ndsm_mean`/`ndsm_std`).
   Il modello usa le statistiche ImageNet come primo layer del backbone, per coerenza con i pesi pre-addestrati.

Numeri attesi per Potsdam `standard` con crop 512 e stride 512: 144 crop per tile (12 × 12), quindi 3168 crop di
train (22 tile), 288 di validation (2 tile) e 14 tile di test. Tempi di preparazione in [ESECUZIONE_E_TEMPI.md](ESECUZIONE_E_TEMPI.md#2-preparazione-dei-dati).

## 6. nDSM come 4° canale (`data.ndsm: true`)

L'nDSM è l'altezza sul terreno. Codifica comune a tutte le fonti: `uint8` in **decimetri**, con saturazione a 25,5 m.

| `data.ndsm_source` | Fonte |
|---|---|
| `official` (default per Potsdam con `auto`) | `1_DSM_normalisation/dsm_potsdam_XX_YY_normalized_<variant>.jpg`, prodotto da ISPRS senza controllo manuale; `data.ndsm_variant: lastools` (default) oppure `ownapproach`; **`ownapproach` non è utilizzabile**: 11 file su 38 sono vuoti (0 byte) già nell'archivio ISPRS (tile 3_13, 4_10, 4_11, 4_13, 5_11, 5_12, 6_8, 6_11, 6_12, 7_7, 7_8) |
| `computed` (default per Vaihingen con `auto`) | calcolato dal DSM assoluto da `rsseg.data.ndsm.compute_ndsm` |

Mediana dell'nDSM per classe (decimetri) su tre tile di Potsdam:

| Tile | Fonte | impervious | building | low veg. | tree | car | clutter |
|---|---|---|---|---|---|---|---|
| 2_10 | `lastools` | 10 | 95 | 10 | 13 | 11 | 10 |
| 2_10 | `ownapproach` | 0 | 93 | 0 | 11 | 10 | 0 |
| 2_10 | `computed` | 8 | 100 | 8 | 20 | 17 | 3 |
| 4_12 | `lastools` | 10 | 127 | 11 | 13 | 11 | 21 |
| 4_12 | `ownapproach` | 0 | 124 | 0 | 9 | 10 | 17 |
| 4_12 | `computed` | 7 | 143 | 8 | 18 | 17 | 24 |

La variante `lastools` ha un offset di circa +1 m sul terreno, mentre `ownapproach` lo mette a 0. Gli edifici
sono coerenti in tutte le fonti. Le mediane degli alberi sono basse: il DSM da image matching coglie poco le
chiome. Un'ablation tra le due varianti ufficiali non è possibile perché `ownapproach` è incompleta: `rsseg prepare
--dry-run` segnala le tile senza nDSM come errore (i file vuoti contano come mancanti). RQ6 confronta quindi
`lastools` con l'nDSM calcolato.

**nDSM calcolato.** Il terreno è stimato con un'apertura morfologica: si prende il minimo del DSM su celle di ~1 m,
poi si applicano un filtro di minimo e uno di massimo con finestra di ~30 m. Tutto ciò che è più stretto della
finestra (edifici, alberi, auto) viene rimosso. La superficie risultante viene smussata e riportata alla
risoluzione originale. È un'approssimazione: gli edifici più larghi di 30 m mantengono parte della loro altezza
nel terreno stimato. Sulla tile 2_10 di Potsdam, l'nDSM calcolato ha correlazione 0,98 con quello ufficiale
(errore medio assoluto 0,57 m). Per confrontare le due fonti sullo stesso dataset si usa
`data.ndsm_source=computed` su Potsdam.

**Uso nel modello.** I crop nDSM sono file a parte (`<stem>_ndsm.png`), quindi lo stesso `data.root` serve sia
gli esperimenti RGB sia quelli RGB+nDSM. Con `data.ndsm: true`, `rsseg train` costruisce il modello a 4 canali
e normalizza l'nDSM con `ndsm_mean`/`ndsm_std` di `stats.json`, salvate nella `config.yaml` del run. I pesi RGB
(ImageNet o DINO) vengono caricati e poi il primo layer (`conv1` o `patch_embed`) viene esteso al 4° canale:
`model.extra_channel_init: mean` (media dei kernel RGB) oppure `zeros` (all'inizio il modello coincide con
quello RGB). Configurazioni: `configs/experiments/maresunet_r18_imagenet_ndsm.yaml`,
`configs/experiments/vaihingen_maresunet_r18_imagenet_ndsm.yaml`.

## 7. Valutazione: etichette piene ed eroded

Ogni tile di val/test viene predetta una volta sola e valutata sia con le etichette piene sia con quelle
**eroded**. Nelle eroded i bordi degli oggetti (qualche pixel) sono neri, cioè ignorati. Il report contiene
`all_classes`/`without_clutter` (piene) e `eroded.all_classes`/`eroded.without_clutter`. MLflow riceve le
metriche con suffisso `_eroded`, e `rsseg report` aggiunge le colonne "mIoU eroded".

Il tool ufficiale ISPRS (`assess_classification_reference_implementation.tar`, `main.cpp`) usa le stesse
definizioni di `rsseg`:

- confusione **assoluta** sommata sui pixel in cui l'etichetta di riferimento è valida (i bordi neri delle eroded
  sono esclusi);
- OA = diagonale / totale;
- per classe precision (*correctness*), recall (*completeness*) e F1;
- le classi assenti dal riferimento sono escluse dalle medie.

Il tool valuta ogni tile due volte, con prefissi `full_reference` e `no_boundaries`. Non calcola l'IoU, che
`rsseg` aggiunge dalla stessa matrice di confusione.

## 8. Pool non etichettati per DINO (`rsseg prepare-unlabeled`)

```
rsseg prepare-unlabeled --source toronto          --raw data/download/Toronto/Images   --out data/unlabeled/toronto
rsseg prepare-unlabeled --source vaihingen_frames --raw data/download/Vaihingen/Images --out data/unlabeled/vaihingen_frames
```

| Sorgente | Frame | Formato | GSD |
|---|---|---|---|
| `toronto` | 13 × 11500 × 7500 | RGB 8 bit | ~15 cm |
| `vaihingen_frames` | 20 × 7680 × 13824 | IR-R-G **16 bit** (dati a 12 bit) | ~8 cm |

- I frame di Vaihingen sono a 16 bit: Pillow li legge tenendo solo il byte alto, con valori ≈ 0–15. Vengono
  quindi letti con `tifffile` e portati a 8 bit con uno stretch per canale ai percentili 0,5–99,5, calcolato su
  un livello della piramide interna del TIFF.
- Crop di 512 non sovrapposti; quelli quasi uniformi vengono scartati. `--max-per-frame N` campiona al massimo N
  crop per frame. `previews/` contiene una miniatura di ogni frame, per controllare i colori.
- Sono immagini prospettiche, non ortofoto: vanno bene per il self-supervised, non per la valutazione.
- `ssl.extra_pools: [data/unlabeled/toronto, ...]` aggiunge i pool ai crop di training del dataset. Le
  configurazioni pronte sono `configs/ssl/dino_r18_toronto.yaml` e `configs/ssl/dino_r18_toronto_vaihingen.yaml`.

**Leakage.** I frame (e il mosaico) di Vaihingen coprono anche le aree di **test** di Vaihingen. Ogni pool
registra la sua città in `source.json`, e `rsseg pretrain` rifiuta un pool della stessa città di
`data.dataset`, a meno che non si imposti `ssl.allow_same_city_pool: true`. In quel caso l'esperimento va
dichiarato transduttivo.

| Pre-training per… | Toronto | Frame di Vaihingen |
|---|---|---|
| esperimenti su Potsdam | ✓ (stesso dominio di colore RGB) | ✓ (colori IR-R-G diversi: variante da confrontare) |
| esperimenti su Vaihingen | ✓ | ✗ (coprono le aree di test) |

## 9. Augmentation

| Fase | Augmentation |
|---|---|
| Supervisionata (default, `data.augment: true`) | flip orizzontale e verticale, rotazioni di 90° (`JointAugment`), applicati in modo identico a immagine (anche l'nDSM) e maschera. Nessuna interpolazione, quindi le etichette restano esatte. |
| Supervisionata (`data.augment: false`, "come la tesi") | nessuna augmentation |
| DINO | multi-crop 2×224 globali (scala 0.14–1 o 0.4–1) + N×96 locali (scala 0.05–0.4), flip, color jitter, grayscale, blur, solarization. Con l'opzione `aerial` le trasformazioni cromatiche sono attenuate, in linea con l'osservazione di MARE. DINO lavora solo su 3 canali. |

## 10. Regime "pochi dati etichettati"

Per la RQ3 la configurazione accetta `data.label_fraction`, per esempio `label_fraction ∈ {0.01, 0.1, 0.25, 1.0}`: il sottoinsieme di crop
etichettati è campionato **per tile** con seed fisso. Il pre-training DINO usa comunque tutte le immagini di
training, senza etichette, più gli eventuali pool del §8.
