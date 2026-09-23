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
> Il mapping con il vecchio ordine è documentato in `rsseg/data/classes.py`.

## 2. Licenza e reperimento

Il dataset **non è redistribuito** in questo repository. Va scaricato dal sito ISPRS
(<https://www.isprs.org/education/benchmarks/UrbanSemLab/>) accettando le condizioni d'uso. I dati
e le maschere derivate (tile, crop `.npy`) non vanno committati: la cartella `data/` sarà in `.gitignore`.

## 3. Protocolli di split

La nuova implementazione supporta due protocolli, selezionabili da configurazione.

### 3.1 `standard` (predefinito, confrontabile con la letteratura)
Split usato comunemente (ad es. da MMSegmentation):

- **train (24 tile)**: 2_10, 2_11, 2_12, 3_10, 3_11, 3_12, 4_10, 4_11, 4_12, 5_10, 5_11, 5_12,
  6_7, 6_8, 6_9, 6_10, 6_11, 6_12, 7_7, 7_8, 7_9, 7_10, 7_11, 7_12
- **test (14 tile)**: 2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13
- **validation**: un sottoinsieme di tile di training **tenuto fuori per intero** (ad es. 2 tile), mai crop
  della stessa tile.
- Alcuni lavori escludono la tile 7_10 per un errore noto nelle etichette: sarà un'opzione di configurazione.

> Da verificare in fase 1 confrontando con la documentazione ufficiale ISPRS e con lo script di MMSegmentation.

### 3.2 `thesis2022` (riproduzione del protocollo della tesi)
- Rimosse: 2_10, 2_11, 2_12, 2_13, 2_14, 3_10, 6_7, 6_13, 7_13, 7_9.
- Le 28 aree restanti sono divise in 22 train e 6 validation. La tesi non elenca quali: la scelta andrà
  fissata e documentata in `configs/data/potsdam_thesis2022.yaml`.
- Serve solo a confrontarsi con i numeri della tesi, non per le conclusioni.

## 4. Pre-processing

1. Conversione della maschera RGB in indice di classe con una **LUT vettorizzata** (niente loop pixel per pixel).
2. Tiling per split, **dopo** lo split per tile, così nessun crop attraversa due split:
   - training: crop 512×512 (o 256×256) con stride configurabile; la sovrapposizione è ammessa **solo dentro il train**;
   - validation/test: **nessun crop pre-calcolato**. Si valuta sull'intera tile con inferenza a finestra
     scorrevole (sliding window) e fusione delle probabilità nelle zone sovrapposte.
3. Salvataggio in formato compatto (PNG per le maschere, PNG o WebP lossless per le immagini) con un
   manifest CSV/Parquet (`tile_id, split, x, y, path_img, path_mask, class_histogram`).
4. Statistiche di normalizzazione calcolate **solo sul train** e salvate nel manifest. In alternativa si usano
   le statistiche ImageNet quando il backbone è pre-addestrato su ImageNet.

## 5. Augmentation

| Fase | Augmentation |
|---|---|
| Supervisionata (default) | flip orizzontale e verticale, rotazioni di 90°, random crop. Le trasformazioni geometriche sono applicate in modo identico a immagine e maschera; la maschera usa interpolazione *nearest*. |
| Supervisionata (ablation "come la tesi") | nessuna augmentation, solo normalizzazione |
| DINO | multi-crop 2×224 globali (scala 0.14–1 o 0.4–1) + N×96 locali (scala 0.05–0.4), flip, color jitter, grayscale, blur, solarization. Con l'opzione `aerial` le trasformazioni cromatiche sono attenuate, in linea con l'osservazione di MARE. |

## 6. Regime "pochi dati etichettati"

Per la RQ3 il DataModule accetta `label_fraction ∈ {0.01, 0.1, 0.25, 1.0}`: il sottoinsieme di crop
etichettati è campionato **per tile** con seed fisso. Il pre-training DINO usa comunque tutte le immagini di
training, senza etichette.
