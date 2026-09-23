# Dataset: ISPRS Potsdam

## 1. Descrizione

Il dataset ISPRS 2D Semantic Labeling Contest – Potsdam copre una città storica tedesca con grandi isolati,
strade strette e un tessuto urbano denso.

| Proprietà | Valore |
|---|---|
| Tile | 38 true orthophoto (TOP) 6000 × 6000 px |
| GSD | 5 cm |
| Canali disponibili | RGB, IRRG, RGBIR, DSM, nDSM (la tesi usa solo **RGB**) |
| Etichette | 6 classi, maschere RGB (anche versione "eroded" senza bordi) |

Classi e codifica (mantenute identiche alla tesi per confrontabilità):

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

## 2. Licenza e reperimento

Il dataset **non è redistribuito** in questo repository. Va scaricato dal sito ISPRS
(<https://www.isprs.org/education/benchmarks/UrbanSemLab/>) accettando le condizioni d'uso. I dati
e le maschere derivate (crop, maschere di tile) non vanno committati: la cartella `data/` è in `.gitignore`.

## 3. Protocolli di split

La nuova implementazione supporta due protocolli, selezionabili da configurazione.

### 3.1 `standard` (predefinito, confrontabile con la letteratura)
Split usato comunemente (ad es. da MMSegmentation):

- **train (24 tile)**: 2_10, 2_11, 2_12, 3_10, 3_11, 3_12, 4_10, 4_11, 4_12, 5_10, 5_11, 5_12,
  6_7, 6_8, 6_9, 6_10, 6_11, 6_12, 7_7, 7_8, 7_9, 7_10, 7_11, 7_12
- **test (14 tile)**: 2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13
- **validation**: un sottoinsieme di tile di training **tenuto fuori per intero** (ad es. 2 tile), mai crop
  della stessa tile.
  Predefinite: 4_12 e 7_11 (`data.val_tiles` per cambiarle).
- Alcuni lavori escludono la tile 7_10 per un errore noto nelle etichette: opzione `data.exclude_7_10: true`.

> Lo split è definito in `src/rsseg/data/splits.py`. Va ancora verificato confrontandolo con la documentazione
> ufficiale ISPRS e con lo script di MMSegmentation.

### 3.2 `thesis2022` (riproduzione del protocollo della tesi)
- Rimosse: 2_10, 2_11, 2_12, 2_13, 2_14, 3_10, 6_7, 6_13, 7_13, 7_9.
- Le 28 aree restanti sono divise in 22 train e 6 validation. La tesi non elenca quali: qui la validation è
  fissata a 3_14, 4_15, 5_13, 6_15, 7_11, 4_11 (`THESIS2022_VAL_DEFAULT`). Il "test" coincide con la validation,
  valutata su tile intere.
- Configurazione pronta: `configs/experiments/thesis2022_maresunet_r18_imagenet.yaml` (crop 256, nessuna augmentation, Adam).
- Serve solo a confrontarsi con i numeri della tesi, non per le conclusioni.

## 4. Pre-processing (`rsseg prepare`)

1. Le tile vengono assegnate agli split **prima** del taglio, così nessun crop attraversa due split.
2. Conversione della maschera RGB in indice di classe con una **LUT vettorizzata**. I colori non in palette diventano `255` (ignorato).
3. Crop:
   - **train**: `crop_size` (512 di default) con `train_stride`. La sovrapposizione è ammessa **solo qui**.
   - **val**: crop non sovrapposti, usati per la validation durante `fit` (selezione del miglior checkpoint).
   - **val/test**: anche la **tile intera** viene registrata (`tiles.csv` + maschera `tiles/<tile>_mask.png`) per la
     valutazione finale a finestra scorrevole con fusione delle probabilità.
   - I crop interamente non etichettati vengono scartati. L'ultimo crop di ogni riga/colonna è allineato al bordo.
4. Formato: PNG per immagini e maschere, `crops.csv` con istogramma delle classi per crop, `split.json`.
5. `stats.json`: media e deviazione standard per canale **solo sul train**. Il modello usa le statistiche ImageNet
   come primo layer del backbone, per coerenza con i pesi pre-addestrati.

## 5. Augmentation

| Fase | Augmentation |
|---|---|
| Supervisionata (default, `data.augment: true`) | flip orizzontale e verticale, rotazioni di 90° (`JointAugment`), applicati in modo identico a immagine e maschera. Nessuna interpolazione, quindi le etichette restano esatte. |
| Supervisionata (`data.augment: false`, "come la tesi") | nessuna augmentation |
| DINO | multi-crop 2×224 globali (scala 0.14–1 o 0.4–1) + N×96 locali (scala 0.05–0.4), flip, color jitter, grayscale, blur, solarization. Con l'opzione `aerial` le trasformazioni cromatiche sono attenuate, in linea con l'osservazione di MARE. |

## 6. Regime "pochi dati etichettati"

Per la RQ3 la configurazione accetta `data.label_fraction`, per esempio `label_fraction ∈ {0.01, 0.1, 0.25, 1.0}`: il sottoinsieme di crop
etichettati è campionato **per tile** con seed fisso. Il pre-training DINO usa comunque tutte le immagini di
training, senza etichette.
