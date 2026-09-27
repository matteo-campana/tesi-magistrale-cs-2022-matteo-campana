# Analisi della tesi

> Documento di riferimento: `legacy/tesi-magistrale/Tesi_Matteo_Campana_1692599_V4.pdf`
> (74 pagine), integrato con il testo della presentazione (`Presentazione Tesi testo.pdf`) e con il codice
> originale nelle cartelle `legacy/image-segmentation-with-swin-transformers/` e `legacy/semantic-segmentation-potsdam/`.

## 1. Scheda

| Voce | Valore |
|---|---|
| Titolo | *Semantic Segmentation for Remote Sensing Images* |
| Autore | Matteo Campana (matricola 1692599) |
| Corso | Laurea Magistrale in Engineering in Computer Science, Sapienza Università di Roma |
| Relatore | Prof. Aurelio Uncini |
| Correlatori | Dr. Simone Scardapane, Dr. Valerio Marsocci (nella tesi scritto "Masocci") |
| Anno accademico | 2021/2022 |
| Dataset | ISPRS 2D Semantic Labeling – **Potsdam** |
| Task | Segmentazione semantica a 6 classi di ortofoto aeree ad altissima risoluzione (5 cm/pixel) |

## 2. Di cosa parla, in una frase

La tesi prende l'architettura **MAResU-Net** (ResNet + U-Net + *Linear Attention Mechanism* nelle skip
connection), già usata dal lavoro **MARE** dei correlatori, e verifica se due modifiche ne migliorano
l'accuratezza su Potsdam:

1. un **collo di bottiglia quantizzato** (codebook VQ-VAE, van den Oord et al. 2017) tra encoder e decoder;
2. un **pre-training self-supervised con DINO** (self-distillation teacher/student, Caron et al. 2021)
   al posto del pre-training supervisionato su ImageNet.

Il risultato sperimentale è **negativo per entrambe**: la configurazione migliore resta la MAResU-Net
"vanilla" con backbone ResNet18 pre-addestrato su ImageNet (test mIoU ≈ 0.604, OA ≈ 0.791).

## 3. Il problema e perché è difficile

La tesi (cap. 1) motiva il lavoro con quattro difficoltà tipiche del telerilevamento urbano VHR:

- **Alta variabilità intra-classe e bassa variabilità inter-classe.** Gli edifici hanno materiali e forme
  diverse, mentre tetti, marciapiedi e strade in cemento si somigliano.
- **Condizioni di ripresa.** Nuvole, foschia, inquinamento e ombre degli edifici alti occludono o
  alterano porzioni dell'immagine.
- **Scarsità di dati etichettati.** A differenza di Cityscapes, ADE20K, COCO-Stuff e PASCAL Context,
  i dataset aerei annotati sono pochi, costosi da acquisire e richiedono esperti per produrre le maschere.
- **Contesto.** Classificare il singolo pixel non basta, serve capire l'oggetto e il suo intorno
  (approccio OBIA). Da qui l'interesse per le reti encoder-decoder con attenzione.

## 4. Il filo logico: perché la tesi è costruita così

Leggere la tesi come una sequenza di scelte chiarisce le motivazioni, anche quelle implicite:

```
Etichette scarse e costose ──► pre-training self-supervised (usare dati NON etichettati)
        │                                   │
        │                        MARE (Marsocci, Scardapane, Komodakis 2021):
        │                        MAResU-Net + OBoW su Vaihingen → migliora la pixel accuracy
        │                                   │
        ▼                                   ▼
Immagini grandi (6000×6000) ──►  attenzione LINEARE O(N) (LAM) invece del dot-product O(N²)
        │                                   │
        ▼                                   ▼
 Continuazione di MARE:     (a) dataset più grande: Vaihingen → Potsdam
                            (b) pre-training più recente: OBoW → DINO (migliore su COCO/ADE20K;
                                funziona anche con backbone ResNet, non solo ViT)
                            (c) quantizzazione esplicita: OBoW quantizza le feature su un
                                vocabolario di "visual words" → idea di rendere la quantizzazione
                                parte dell'architettura con un codebook VQ-VAE (ispirato anche
                                dal successo di VQ-GAN, "Taming Transformers")
```

### 4.1 Perché MAResU-Net
- **ResNet come encoder**: backbone standard con pesi ImageNet disponibili, utile per il transfer learning.
- **Struttura U-Net**: le skip connection recuperano i dettagli spaziali persi nel downsampling, cosa
  indispensabile per contorni precisi.
- **LAM (Li et al. 2020)**: sostituisce `softmax(QKᵀ)` con un'approssimazione di Taylor al primo ordine
  su Q e K normalizzati L2, riducendo il costo da O(N²) a O(N) sui pixel. Sui canali (C ≪ N) si usa invece
  una normale channel attention (CAM). Su immagini ad alta risoluzione è ciò che rende l'attenzione praticabile.
- **Continuità con MARE**: tenere la stessa architettura rende i risultati confrontabili con il lavoro dei
  correlatori.

### 4.2 Perché la Vector Quantization
- La tesi (cap. 2.4) sostiene che molte modalità (linguaggio, parlato, oggetti nelle immagini) sono
  naturalmente **discrete**: un codebook costringe il bottleneck a rappresentare solo le caratteristiche
  essenziali e fa da regolarizzatore.
- Il collegamento con OBoW è diretto: OBoW assegna le feature al "visual word" più vicino di un vocabolario,
  e VQ fa esattamente questo con un codebook appreso.
- Ipotesi di ricerca dichiarata (cap. 1): *"il layer di quantizzazione interposto tra encoder e decoder può
  aumentare l'accuratezza del modello"*.

### 4.3 Perché DINO
- La self-distillation (teacher aggiornato come EMA dello student, centering e sharpening dell'output del
  teacher, multi-crop globale/locale) non ha bisogno di etichette. Si può quindi pre-addestrare l'encoder
  sulle sole immagini di Potsdam.
- DINO ed EsViT risultavano migliori di OBoW nei benchmark e mostravano risultati promettenti anche su ResNet.
- La tesi riprende da MARE un'osservazione: le augmentation "forti" aiutano su ImageNet/COCO ma **non
  necessariamente** sulle immagini telerilevate (§3.2). È un punto che resta aperto.

### 4.4 Perché ResNet18/34 e non backbone più grandi
- Le risorse di calcolo erano limitate: il codice gira su Google Colab e usa l'autoscaling della batch size
  di PyTorch Lightning.
- Nelle conclusioni c'è una motivazione esplicita di **efficienza energetica**: i modelli piccoli sono adatti
  all'inferenza on-board, in tempo reale su droni o satelliti.

## 5. Impianto sperimentale (come descritto nella tesi)

### 5.1 Dati (cap. 4.1)
- Potsdam: 38 tile 6000×6000 RGB (TOP), GSD 5 cm, 6 classi:

  | id (codice) | Classe | RGB |
  |---|---|---|
  | 0 | Clutter/Background | (255, 0, 0) |
  | 1 | Impervious surfaces | (255, 255, 255) |
  | 2 | Car | (255, 255, 0) |
  | 3 | Building | (0, 0, 255) |
  | 4 | Low vegetation | (0, 255, 255) |
  | 5 | Tree | (0, 255, 0) |

- Le tile vengono tagliate in crop 256×256 (ridimensionati a 224×224 dal dataloader). I bordi in eccesso
  vengono scartati.
- Sono state **rimosse 10 aree** perché ritenute troppo omogenee: 2_10, 2_11, 2_12, 2_13, 2_14, 3_10, 6_7, 6_13, 7_13, 7_9.
- Delle 28 aree rimaste, 22 vanno in training e 6 in validation (~11.000 crop di training).
- Pre-training DINO: 2 crop globali 224 (scala 0.14–1) e crop locali 96 (scala 0.05–0.4), flip, color jitter,
  grayscale, blur, solarization.
- Fine-tuning: nessuna augmentation, solo normalizzazione.

### 5.2 Griglia di esperimenti: 2 × 2 × 2 = 8 modelli

| Asse | Valori |
|---|---|
| Architettura | MAResU-Net, MAResU-Net-VQ |
| Backbone | ResNet18, ResNet34 |
| Pre-training | ImageNet (supervisionato), DINO (self-supervised su Potsdam) |

Iperparametri (tab. 4.1–4.2): Adam, lr 1e-3, StepLR (step 15, γ 0.4), 100 epoche, batch 32.
La loss è CE per i modelli base e "MSE + VQ loss" per i modelli VQ. DINO usa uno scheduler coseno,
weight decay 0.04→0.4, τₛ = 0.1 e τₜ = 0.04→0.07.

### 5.3 Risultati di test (tab. 4.5)

| Modello | Backbone | Pre-train | OA | F1 | mIoU |
|---|---|---|---|---|---|
| MAResU-Net | R18 | ImageNet | 0.791 | **0.722** | **0.604** |
| MAResU-Net | R34 | ImageNet | **0.791** | 0.722 | 0.604 |
| MAResU-Net | R18 | DINO | 0.741 | 0.657 | 0.530 |
| MAResU-Net | R34 | DINO | 0.739 | 0.653 | 0.525 |
| MAResU-Net-VQ | R18 | ImageNet | 0.750 | 0.668 | 0.543 |
| MAResU-Net-VQ | R34 | ImageNet | 0.744 | 0.660 | 0.534 |
| MAResU-Net-VQ | R18 | DINO | 0.702 | 0.593 | 0.465 |
| MAResU-Net-VQ | R34 | DINO | 0.703 | 0.595 | 0.467 |

Per classe (test mIoU, miglior modello), Building è la classe migliore (0.80), seguita da Car (0.73),
Impervious (0.72), Tree (0.62) e Low vegetation (0.61). Background/Clutter è la peggiore (0.15).

### 5.4 Conclusioni dell'autore
1. VQ e DINO **peggiorano** tutte le metriche e i loro effetti si sommano (VQ+DINO è il caso peggiore).
2. ResNet18 ≈ ResNet34: meglio il modello più piccolo, per costo ed energia.
3. Clutter è la classe più rumorosa. Tree e Low vegetation si confondono (bordi sfumati, ombre).
4. Validation (mIoU ≈ 0.96) e test (≈ 0.60) sono molto lontani. L'autore lo attribuisce a overfitting.
5. DINO perde contro ImageNet. L'ipotesi è che ~11k crop siano troppo pochi rispetto ai milioni di ImageNet.
6. Sviluppi futuri: transformer, attenzione efficiente, paradigma teacher-student.

## 6. Lettura critica: che cosa i risultati permettono davvero di concludere

Confrontando la tesi con il codice originale (dettagli in [`REVISIONE_CODICE_ORIGINALE.md`](REVISIONE_CODICE_ORIGINALE.md))
emergono diversi fattori, **alternativi alle spiegazioni proposte nella tesi**, che possono aver prodotto
il risultato negativo. Il punto per la reimplementazione è che le ipotesi di ricerca **non sono state
confutate in modo conclusivo**: vanno ritestate con un protocollo corretto.

| Osservazione nella tesi | Spiegazione proposta | Fattori alternativi trovati nel codice/protocollo |
|---|---|---|
| Clutter quasi mai riconosciuto | Eterogeneità della classe | `CrossEntropyLoss(ignore_index=0)` e la classe 0 è proprio Clutter: il modello **non riceve mai gradiente** per quella classe. |
| Validation ≫ test | Overfitting | L'overfitting si vedrebbe come train ≫ val, non val ≫ test. Un salto 0.96 → 0.60 fa pensare a una **validation non indipendente**: il notebook di preparazione (Vaihingen) produce crop sovrapposti con stride 128, e per Potsdam lo script non è presente, quindi una sovrapposizione tra split non si può escludere o a metriche calcolate in modo diverso (media per batch con `nanmean` invece di una matrice di confusione globale). |
| DINO < ImageNet | Dataset troppo piccolo | L'implementazione si discosta da DINO in modo sostanziale: non c'è projection head, la loss è calcolata sull'output denso del decoder appiattito su **tutto il batch** con un'unica softmax (mescola i campioni), i crop locali sono a 224 e non a 96, la temperatura del teacher è fissa, manca il weight decay schedulato e il teacher non nasce come copia dello student. |
| VQ peggiora | La VQ non aiuta | Il codebook è solo sul bottleneck e le skip connection lo aggirano. La variante EMA ricrea `nn.Parameter` a ogni step. La loss è CE + VQ "clippata" in modo euristico (la tesi dice MSE + VQ). Il default `base_model=resnet18(pretrained=True)` viene valutato **una volta sola all'import**, quindi più istanze (teacher e student in DINO-VQ) **condividono lo stesso encoder**: congelare il teacher congela anche lo student. |
| R18 ≈ R34 | Il modello piccolo basta | Nel codice disponibile esistono ResNet18 e ResNet50 ma non ResNet34, quindi il codice che ha prodotto le tabelle non è con certezza quello nel repository. |

Altri punti metodologici:
- **Split non standard.** Le aree rimosse includono 2_13, 2_14, 6_13 e 7_13, che nello split usato più comunemente in
  letteratura (ad es. MMSegmentation) sono tile di **test**. I numeri non sono quindi confrontabili con lo stato dell'arte.
- **Il test set non è descritto.** Il testo parla di 22 aree di training e 6 di validation, poi riporta "Test Scores"
  senza dire da dove vengano.
- **Nessuna normalizzazione degli input** nel dataset di segmentazione (solo `/255`), mentre la tesi la dichiara.
  Per un backbone ImageNet la normalizzazione con media e deviazione standard di ImageNet conta.
- **Train DataLoader senza shuffle.**
- **Una sola esecuzione per configurazione, senza seed**: le differenze di 1–2 punti non sono significative.
- **Refusi nelle citazioni.** "EsViT" viene attribuito al paper DeiT ("Training data-efficient image
  transformers") e il titolo di EsViT viene associato a DINO. Il cognome del correlatore è Marsocci.

## 7. Domande di ricerca da preservare nella reimplementazione

Queste sono le domande che la tesi voleva porre e che il nuovo codice deve permettere di ritestare in modo
pulito e riproducibile:

- **RQ1.** Su Potsdam, con un protocollo standard, quanto rende MAResU-Net (LAM) rispetto a una U-Net/ResUNet
  senza attenzione, a parità di backbone?
- **RQ2.** Un bottleneck quantizzato (VQ, con codebook sano cioè con perplexity alta e pochi codici morti)
  migliora la segmentazione? Conta dove viene inserito (solo bottleneck o anche le skip connection)?
- **RQ3.** Un pre-training DINO **fedele all'originale** sull'encoder, con dati non etichettati di dominio,
  batte o eguaglia ImageNet? Come cambia al variare della quantità di dati etichettati (1%, 10%, 100%)?
  È lo scenario che motiva davvero il self-supervised.
- **RQ4.** Le augmentation "forti" di DINO sono adatte alle immagini aeree (osservazione ereditata da MARE)?
- **RQ5.** Quali sono i trade-off accuratezza/costo (parametri, FLOPs, latenza) tra R18, R34 e R50?
  Riprende la motivazione energetica delle conclusioni.

Estensioni aggiunte nella reimplementazione (non presenti nella tesi):

- **RQ6.** L'altezza sul terreno (nDSM) come 4° canale di input ("early fusion") migliora la segmentazione,
  in particolare per building vs impervious surfaces e tree vs low vegetation? Il risultato cambia se l'nDSM
  è quello ufficiale o quello stimato dal DSM?
- Ogni RQ viene verificata anche su **Vaihingen** (secondo benchmark), e la RQ3 anche con pool non etichettati
  di altre città (Toronto, frame di Vaihingen). Dettagli in [DATASET.md](DATASET.md).

## 8. Bibliografia essenziale

- Marsocci, Scardapane, Komodakis. *MARE: Self-Supervised Multi-Attention REsu-Net for Semantic Segmentation
  in Remote Sensing.* Remote Sensing 13(16), 3275, 2021.
- Li, Zheng, Duan, Su, Zhang. *Multistage Attention ResU-Net for Semantic Segmentation of Fine-Resolution
  Remote Sensing Images.* IEEE GRSL 19, 2022.
- Li, Su, Duan, Zheng. *Linear Attention Mechanism: An Efficient Attention for Semantic Segmentation.* arXiv:2007.14902.
- van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation Learning.* NeurIPS 2017.
- Caron et al. *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV 2021.
- Gidaris et al. *OBoW: Online Bag-of-Visual-Words Generation for Self-Supervised Learning.* CVPR 2021.
- Li et al. *Efficient Self-supervised Vision Transformers for Representation Learning (EsViT).* arXiv:2106.09785.
- He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Ronneberger et al. *U-Net.* MICCAI 2015.
