# Come lanciare test ed esperimenti, e quanto durano

Questa guida spiega come lanciare i test automatici, la preparazione dei dati e gli esperimenti (singoli o in
campagne), con i tempi di esecuzione. Per installazione, memoria GPU e MLflow vedi [GPU_LOCALE.md](GPU_LOCALE.md);
per i dati [DATASET.md](DATASET.md).

**Macchina delle misure:** NVIDIA RTX 4060 8 GB, 16 GB di RAM, Windows 11 nativo, Python 3.11, PyTorch 2.14 +
CUDA 13.2, dati su SSD. Su un'altra macchina i tempi cambiano in proporzione alla velocità della GPU. Su Windows
nativo l'avvio dei worker del DataLoader aggiunge circa un minuto a ogni run.

Legenda: **misurato** = cronometrato su questa macchina; **stima** = calcolato dai tempi misurati (formula in §4.1).

## 1. Test automatici (pytest)

```bash
pytest -q                          # tutti gli 80 test
pytest -q -k "not end_to_end"      # esclude i 3 test end-to-end più lenti
pytest -q tests/test_isprs.py      # un solo file
pytest -q --durations=10           # mostra i test più lenti
ruff check src tests && ruff format --check src tests   # stile e lint (pochi secondi)
```

I test non usano i dati ISPRS reali: generano tile sintetiche in cartelle temporanee. Non toccano `runs/`, `data/`
né `mlflow.db`.

| Cosa | Tempo (misurato) |
|---|---|
| Tutti i test, con GPU | 1 min 53 s |
| Tutti i test, solo CPU (`CUDA_VISIBLE_DEVICES=""`, come in CI) | 2 min 6 s |
| Senza i test end-to-end (`-k "not end_to_end"`) | 39 s |

| File | Contenuto | Tempo (misurato) |
|---|---|---|
| `tests/test_cli.py` | parser, comandi `backbones` e `report`, tabelle | 9 s |
| `tests/test_data.py` | classi, split, crop, dataset, augmentation | 10 s |
| `tests/test_isprs.py` | layout reali, etichette corrotte, Vaihingen, nDSM, eroded, pool, training con nDSM | 25 s |
| `tests/test_models.py` | equivalenza con torchvision, LAM, VQ, encoder a 4 canali | 20 s |
| `tests/test_training.py` | loss, metriche, scheduler, DINO, finestra scorrevole, top-k, training end-to-end | 30 s |
| `tests/test_campaign.py` | campagne con runner finto + una campagna reale con 3 sottoprocessi | 74 s |

I tre test più lenti: la campagna reale (60 s), il training + DINO end-to-end (13 s) e il training con nDSM (10 s).
Ogni file lanciato da solo costa qualche secondo in più per l'import di PyTorch e Keras.

## 2. Preparazione dei dati

| Comando | Tempo (misurato) | Spazio |
|---|---|---|
| `rsseg prepare -c configs/base.yaml --dry-run` (38 tile) | ~40 s | — |
| `rsseg prepare -c configs/experiments/maresunet_r18_imagenet_ndsm.yaml` (Potsdam + nDSM, 4 worker) | 2 min 40 s | 1,5 GB |
| `rsseg prepare -c configs/experiments/vaihingen_maresunet_r18_imagenet_ndsm.yaml` | 26 s | 0,5 GB |
| `rsseg prepare-unlabeled --source toronto …` (13 frame, 4433 crop) | pochi minuti | 1,9 GB |
| `rsseg prepare-unlabeled --source vaihingen_frames …` (20 frame) | stima 5–10 min | stima 3–4 GB |

Le campagne lanciano da sole questi comandi quando i dati mancano. Per esempio `rq6_ndsm` prepara
`data/potsdam_ndsm_computed` (stima ~3 min) e `thesis2022` prepara
`data/potsdam_thesis2022`.

## 3. Esperimenti singoli

```bash
rsseg train    -c configs/experiments/maresunet_r18_imagenet.yaml          # training + test
rsseg pretrain -c configs/ssl/dino_r18.yaml                                # DINO
rsseg evaluate -c configs/experiments/maresunet_r18_imagenet.yaml --weights runs/<run>/best.weights.h5
rsseg benchmark -c configs/experiments/maresunet_r18_imagenet.yaml         # VRAM e velocità, ~30 s
```

Un run di training ha tre parti:

1. **avvio**, circa 1–2 minuti (misurato): import, worker del DataLoader, pesi ImageNet, MLflow;
2. **epoche**: al massimo `train.epochs` = 100, ma l'early stopping si ferma dopo 20 epoche senza miglioramenti
   della `val_miou`;
3. **valutazione finale** su tutte le tile di test, con finestra scorrevole.

### 3.1 Tempi misurati per epoca e per valutazione

| Esperimento | Passi/epoca | Epoca (misurato) | Valutazione test (misurato) |
|---|---|---|---|
| MAResU-Net R18 (Potsdam) | 396 + 36 val | 2,8 min | 7 min (14 tile × 0,5 min) |
| MAResU-Net R18 + nDSM | 396 + 36 | 2,8 min | 7 min |
| MAResU-Net-VQ R18 | 396 + 36 | 2,7 min | ~7 min (stima) |
| MAResU-Net R34 | 396 + 36 | 3,2 min | 7 min |
| MAResU-Net Swin-T (batch 4) | 792 + 72 | 4,7 min | ~7 min (stima) |
| MAResU-Net ViT-S/16 (batch 4) | 792 + 72 | 4,3 min | 6,4 min |
| MAResU-Net ViT-B/16 (batch 4) | 792 + 72 | **29,6 min** | 9 min |
| MAResU-Net R18 (Vaihingen) | 113 + 6 | 0,8 min (stima dal passo misurato) | ~1 min (17 aree) |
| DINO R18 (Potsdam) | 99 | 0,7 min | — |
| DINO R18 + pool Toronto | 237 | 1,7 min | — |
| DINO R34 | 99 | 1,2 min | — |
| DINO ViT-S/16 | 99 | 0,8 min | — |

Il passo di training costa circa 0,40 s per R18 (20 immagini 512²/s): il collo di bottiglia è la GPU, non il
caricamento dei dati. ViT-B/16 è 7 volte più lento di R18. Il suo picco di VRAM (7,1 GB) è al limite degli 8 GB e
probabilmente usa anche la memoria condivisa di sistema. Su questa GPU non è praticabile con 100 epoche: vedi §5.

### 3.2 Durata stimata di un run

| Esperimento | Tipico (~60 epoche) | Massimo (100 epoche) |
|---|---|---|
| R18, R18 + nDSM, U-Net R18, R18 da DINO (Potsdam) | ~3,0 h | ~4,8 h |
| VQ R18 | ~2,9 h | ~4,7 h |
| R34, VQ R34 | ~3,4 h | ~5,5 h |
| Swin-T | ~4,9 h | ~8,0 h |
| ViT-S/16 da DINO | ~4,4 h | ~7,3 h |
| ViT-B/16 | **~30 h** | **~50 h** |
| R18 su Vaihingen | ~0,9 h | ~1,4 h |
| `thesis2022` (senza early stopping, crop 256) | ~4,8 h | ~4,8 h |
| R18 con `label_fraction` 0,01 / 0,1 / 0,25 | ~3,0 h | ~4,8 h |
| DINO R18 / R18 + Toronto / R34 / ViT-S (100 epoche fisse) | ~1,2 / 2,8 / 2,0 / 1,3 h | uguale |
| `rsseg benchmark` | ~0,5 min | ~0,5 min |

Con poche etichette le epoche hanno comunque 396 passi: i crop etichettati vengono ripetuti per dare a ogni
frazione lo stesso numero di aggiornamenti dei pesi (`data.full_length_epochs`, vedi [DATASET.md](DATASET.md) §10).
Con meno etichette l'early stopping può però intervenire prima.

## 4. Campagne

```bash
rsseg campaign configs/campaigns/all.yaml --dry-run          # piano, disco, tempo stimato (~12 s)
rsseg campaign configs/campaigns/rq1_attention.yaml --quick  # prova veloce (vedi §4.3)
rsseg campaign configs/campaigns/rq1_attention.yaml          # esecuzione vera (riprende se interrotta)
rsseg campaign configs/campaigns/transformers.yaml --only "maresunet_swin_t_*" "maresunet_vit_s_16_*"
```

Dettagli sul funzionamento (store condiviso, ripresa, errori, report) in [GPU_LOCALE.md](GPU_LOCALE.md#6-training-ed-esperimenti).

### 4.1 Durata stimata per campagna (3 seed)

Formula: durata di un run = avvio (~2 min) + epoche × tempo per epoca (§3.1) + valutazione. "Tipico" suppone che
l'early stopping si fermi intorno a 60 epoche (con lo scheduler `step` il learning rate scende del 60% ogni 15
epoche); "massimo" usa tutte le 100 epoche. Sono stime: `rsseg campaign … --dry-run` mostra la stima di ogni passo e
l'ora di fine prevista, ricalcolate con le velocità misurate dai run già completati (§4.4).

| Campagna | Contenuto | Passi | Tipico | Massimo |
|---|---|---|---|---|
| `rq5_efficiency` | 10 benchmark | 10 | ~6 min | ~6 min |
| `vaihingen` | 3 config × 3 seed | 9 | ~8 h | ~13 h |
| `thesis2022` | 1 config × 3 seed (+ prepare) | 3 | ~15 h | ~15 h |
| `rq1_attention` | 2 config × 3 seed | 6 | ~18 h | ~29 h |
| `rq4_aerial` | 2 DINO + 4 config × 3 seed | 14 | ~32 h | ~48 h |
| `rq6_ndsm` | 3 config × 3 seed (+ 1 prepare) | 9 | ~27 h | ~44 h |
| `rq3_low_label` | 2 DINO + 12 config × 3 seed | 38 | ~112 h | ~177 h |
| `thesis_grid` | 2 DINO + 8 config × 3 seed | 26 | ~79 h | ~125 h |
| `transformers` | 1 DINO + 3 config × 3 seed | 10 | ~120 h (~30 h senza ViT-B) | ~195 h (~47 h senza ViT-B) |
| `all` | tutto, run in comune eseguiti una volta | 105 | **~16 giorni** (~12 senza ViT-B) | ~25 giorni (~19 senza ViT-B) |

I run in comune tra campagne (ad esempio `maresunet_r18_imagenet`, presente in RQ1, RQ3, RQ6 e nella griglia)
vengono addestrati una volta sola: lanciare prima `rq1_attention` accorcia le campagne successive.

### 4.2 Ordine consigliato

Dalle campagne che rispondono prima alle domande principali a quelle più lunghe:

1. `rq5_efficiency` (minuti): conferma memoria e velocità sulla tua macchina.
2. `rq1_attention` (~18 h): baseline MAResU-Net R18, riusata da quasi tutte le altre.
3. `vaihingen` (~8 h) e `rq6_ndsm` (~27 h): secondo dataset e nDSM.
4. `rq3_low_label` (~112 h) e `rq4_aerial` (~32 h): DINO; condividono `dino_r18` e `maresunet_r18_dino`.
5. `thesis_grid` (~79 h, meno i run già fatti) e `thesis2022` (~15 h): confronto con la tesi.
6. `transformers` (~30 h senza ViT-B): conviene escludere ViT-B/16 con `--only` (§5).

Per avere prima tutti i risultati con un seed e poi aggiungere gli altri: `--seeds 0`, poi `--seeds 1,2`. La campagna
salta i run già completati.

### 4.3 Prove veloci (`--quick`)

`--quick` esegue ogni passo in miniatura: 1 epoca da 20 passi, 5 passi di validation, 1 tile di test e DINO con 1
epoca da 10 passi. Scrive in `runs/campaigns-quick/`, che non si mescola mai con i risultati veri. Serve a
verificare che una campagna parta e arrivi in fondo prima di lanciarla davvero.

| Prova | Tempo |
|---|---|
| Un run di training `--quick` | ~2–3 min (misurato: 2 min 49 s prima dell'ottimizzazione della valutazione) |
| `rq5_efficiency --quick` | ~5 min (misurato) |
| `rq6_ndsm --quick --seeds 0` | ~8 min + 1 prepare (~3 min) (stima) |
| `all --quick --seeds 0` | ~1,5–2 h (stima) |

### 4.4 Seguire una campagna in corso

La console stampa una riga all'inizio (▶) e una alla fine (✓ o ✗) di ogni passo. Mentre il passo gira, sotto la
riga ▶ c'è un riquadro con tre barre di avanzamento, aggiornato ogni secondo, per esempio:

```
[4/30] train maresunet_r18_imagenet__s0
epoch    ━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━  50% epoch 12/100 · batch 198/396 · 2m24s per epoch
step     ━━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━  22% 35m12s elapsed · ~2h02m left · ends ~21:40
campaign ━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━   5% step 4/30 · ~2d09h left · ends ~Thu 01/10 07:15
  val_miou 0.7120 · best 0.7180 at epoch 10 · stops at epoch 30 unless it improves
  step: 50m48s if it stops at epoch 30, 3h40m with all 100 epochs (test evaluation ~7m08s included)
  log: runs/campaigns/store/runs/maresunet_r18_imagenet__s0/log.txt
```

| Elemento | Significato |
|---|---|
| `[4/30]` | passo corrente e passi da eseguire in questo lancio (quelli già completati non contano) |
| barra `epoch` | batch dell'epoca corrente; `validation` durante la validazione di fine epoca. Durante la valutazione finale diventa la barra `tiles` (`test evaluation · tile 5/14`) |
| barra `step` | tempo trascorso e **proiezione** del tempo rimasto del passo, con l'ora di fine prevista |
| barra `campaign` | tempo rimasto di tutta la campagna (passo corrente + stima dei passi successivi) e ora di fine prevista |
| `val_miou … best …` | ultima `val_mIoU` e migliore finora (il checkpoint `best.weights.h5` è quello della migliore); l'epoca in cui l'early stopping fermerebbe il run se non migliora più |
| `step: A if …, B with all …` | intervallo del passo: A se l'early stopping si ferma appena possibile, B se il run arriva a tutte le epoche |
| `starting: …` | primi 1–2 minuti: import, worker del DataLoader, modello |
| `no progress for …` | nessun aggiornamento da 15 minuti: controlla il log, il processo potrebbe essere bloccato |

**Come vengono calcolate le proiezioni** (`src/rsseg/eta.py`):

- durata di un training = avvio + epoche previste × durata di un'epoca + valutazione;
- **durata di un'epoca**: megapixel di training e validazione dell'epoca × secondi per megapixel dell'encoder;
- **epoche previste**: con early stopping, la media delle epoche dei run già finiti (60 prima del primo) e
  comunque non meno dell'epoca in cui l'early stopping potrebbe fermare il run in corso; senza early stopping,
  `train.epochs`;
- **valutazione**: megapixel delle tile di test × secondi per megapixel;
- DINO: epoche × immagini per epoca × secondi per immagine.

I valori di partenza sono misurati su RTX 4060: per R18 su Potsdam 144 s per epoca e 0,5 min per tile 6000².
Vengono sostituiti dalle misure reali: il passo in corso aggiorna subito la stima dei passi dello stesso encoder, e
ogni passo completato salva le sue velocità in `outcome.json` (campo `timing`), usate dalle stime successive, anche
di altre campagne. Le stime migliorano quindi dopo il primo run completato. `--dry-run` mostra la stima di ogni passo
(colonna `estimate`), il totale e l'ora di fine prevista.

Il riquadro legge `progress.json`, che il training scrive nella cartella del passo. Per vedere tutto il dettaglio
(metriche di ogni epoca, valutazione tile per tile), in un secondo terminale:

```bash
tail -f runs/campaigns/store/runs/maresunet_r18_imagenet__s0/log.txt   # Ctrl+C chiude solo tail
mlflow ui --backend-store-uri sqlite:///mlflow.db                        # curve su http://127.0.0.1:5000
```

**Fermare e riprendere.** Ferma con Ctrl+C e aspetta che torni il prompt; per riprendere rilancia lo stesso comando
(con `--clean-failed` cancella anche la cartella del tentativo interrotto). Il passo interrotto riparte da zero.
Se invece chiudi la finestra di Git Bash, il processo Python del training può restare attivo in background: prima
di rilanciare controlla con `nvidia-smi` che la GPU sia libera ed eventualmente chiudilo con
`taskkill //F //PID <pid>`.

## 5. Ridurre i tempi

| Leva | Come | Effetto |
|---|---|---|
| Meno seed all'inizio | `--seeds 0` | Tempi ÷ 3; gli altri seed si aggiungono dopo |
| Early stopping più aggressivo | `train.early_stopping_patience=10` nella lista `set:` dei run della campagna (o `--set` di `rsseg train`) | Meno epoche; rischio di fermarsi prima di un calo del learning rate |
| Meno epoche | `train.epochs=60` in `set:` (o `--set`) | Tempo massimo −40% |
| Escludere ViT-B/16 | `--only` sugli altri run di `transformers` | −90 h |
| Valutazione più rada | `eval.stride=512` in `set:` (o `--set`) | Valutazione ~4× più veloce, ma senza sovrapposizione tra le finestre (numeri non confrontabili con gli altri run) |
| Linux o WSL2 | vedi [GPU_LOCALE.md](GPU_LOCALE.md) | Avvio dei worker più rapido |

Durante una campagna evita di usare la GPU per altro: un secondo processo sulla GPU rallenta entrambi e può
causare errori di memoria. Una campagna interrotta (Ctrl+C, riavvio) riprende dal primo passo non completato;
il passo interrotto riparte da zero.
