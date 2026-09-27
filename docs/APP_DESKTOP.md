# Applicazione desktop `rsseg-gui`

L'applicazione Qt (PySide6) serve a preparare i dati, configurare e lanciare training e pre-training, seguirne
l'avanzamento, metterli in pausa e riprenderli, controllare la GPU, analizzare i risultati e fare inferenza con i
modelli addestrati. Non sostituisce la riga di comando: ogni azione lancia un comando `rsseg`, che resta
utilizzabile da solo. Per l'architettura del pacchetto vedi [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Installazione e avvio

```bash
pip install -e ".[gui]"          # PySide6, pyqtgraph, nvidia-ml-py (NVML), psutil
cd <cartella del progetto>        # quella con configs/, data/, runs/
rsseg-gui                         # oppure: python -m rsseg.gui --root <cartella>
```

La cartella del progetto è quella da cui partono i job: i percorsi relativi delle configurazioni (`data/…`,
`runs/…`) si riferiscono a essa. L'app scrive il proprio stato in `.gui/` (ignorata da git).

Su Linux senza display (server, CI) l'app gira con `QT_QPA_PLATFORM=offscreen`; servono le librerie di sistema
di Qt (`libegl1`, `libgl1`, `libxkbcommon0`, `libfontconfig1`, `libdbus-1-3`).

## 2. Le sezioni

| Sezione | Cosa si fa |
|---|---|
| **Panoramica** | Stato della GPU, job attivi con le loro barre, run **da riprendere**, dati preparati, ultimi risultati |
| **Configurazione** | Scelta di un YAML, modifica di ogni impostazione in un modulo generato dal codice, stima della durata, «Entra in 8 GB?» (benchmark della VRAM), avvio di training, pre-training DINO, preparazione e valutazione, salvataggio di un nuovo YAML |
| **Training e job** | Coda dei job, fasi, barre, tempi, curve di loss e mIoU (con la linea della ripresa), uso della GPU durante il job, log, risultato, pausa / ripresa / stop |
| **Campagne** | Piano (dry-run) con passi, disco e tempi stimati, avvio e ripresa, stato di ogni passo, report finale |
| **Dati** | Tile sintetiche, verifica e preparazione di Potsdam/Vaihingen (tabella di file, split e anomalie per tile), pool non etichettati per DINO, visualizzatore dei crop con maschera sovrapposta |
| **Risultati** | Tutti i run: metriche per classe (piene ed eroded), matrice di confusione, curve, GPU, predizioni, configurazione; confronto di più run (media ± std sui seed) con esportazione Markdown/CSV/PNG |
| **Inferenza** | Scelta del modello e del checkpoint, immagini o tile di uno split, parametri della finestra scorrevole, visualizzatore sincronizzato, confronto tra due modelli, esportazione |
| **GPU** | Rilevamento, verifica di PyTorch, utilizzo, memoria, temperatura, potenza, frequenza, ventola, grafici degli ultimi 10 minuti, processi |

## 3. Barre di avanzamento e tempi

Ogni job mostra lo stesso pannello:

```
Training maresunet_r18_imagenet  [in esecuzione]                 [⏸ Pausa] [⏹ Stop]
✔ Avvio → ● Training → ○ Valutazione test → ○ Report
Epoca   ███████████░░░░░░░░░  23/100
Batch   ██████████████░░░░░░  812/1150
Totale  █████░░░░░░░░░░░░░░░  27 %
Trascorso 1 h 12 min · Rimanente 2 h 10 min – 3 h 05 min · Fine prevista 18:10 – 19:05
GPU 97 % · VRAM 6,1/8,0 GB · 71 °C · 108 W
val_miou migliore 0.7412 (epoca 19) · early stopping: 4/20 epoche senza miglioramenti
```

- **Fasi**: dipendono dal comando (training: avvio, training, valutazione test, report; inferenza: caricamento del
  modello, inferenza, salvataggio; preparazione: avvio, tile, manifest; campagna: pianificazione, dati,
  esperimenti, report).
- **Barre**: epoca e batch (training, DINO), tile (valutazione), tile o frame (preparazione), immagini e finestre
  (inferenza), passi della campagna più le barre del passo in corso. «Totale» = trascorso / (trascorso + rimanente).
- **Trascorso**: solo il tempo di lavoro. Le pause sono escluse, il lavoro prima di una ripresa è incluso.
- **Rimanente**: prima delle misure viene da `rsseg.eta` (tempi misurati sulla RTX 4060, ricalibrati sui passi
  delle campagne); poi dalla durata media delle epoche, delle tile o delle immagini del job stesso. Con early
  stopping è un intervallo: dal caso in cui il training si ferma appena possibile (epoca migliore + patience) a
  quello in cui fa tutte le epoche. In coda è mostrata la durata stimata.
- Un job che non aggiorna l'avanzamento per 15 minuti viene segnalato in arancione («controlla il log»).
- Nella barra di stato in basso: numero di job attivi, ora prevista di fine, GPU.

## 4. Suggerimenti al passaggio del mouse

Ogni impostazione della configurazione, ogni opzione dei comandi, ogni metrica e ogni valore della GPU ha un
suggerimento in italiano: cosa fa, valori ammessi, effetti pratici (VRAM, batch effettivo, tempo), valore
predefinito, **file da cui arriva il valore attuale** (per esempio `configs/base.yaml` o il file dell'esperimento)
e dove approfondire. Selezionando un campo, la stessa spiegazione compare nel pannello «Guida».

I testi sono in `src/rsseg/gui/help_it.yaml`. Un test (`tests/test_gui_core.py`) fallisce se un campo di
`rsseg.config` non ha il suo testo: aggiungendo un'impostazione al codice va aggiunta anche la sua spiegazione.

## 5. Pausa, chiusura dell'app e ripresa

- **Pausa**: il job finisce il batch in corso, salva un **checkpoint di ripresa** in `<run>/resume/` e termina,
  liberando la GPU. Durante la validazione la pausa avviene a fine epoca; durante la valutazione finale, tra una
  tile e l'altra.
- **Riprendi**: il job riparte dallo stesso batch, nello **stesso run MLflow**, con gli stessi risultati di un
  run senza interruzioni (verificato dai test su training e DINO). Prima di riprendere l'app mostra da dove
  riparte, quanto lavoro è già stato fatto e se sono cambiati codice (commit git) o librerie.
- **Stop**: termina senza possibilità di ripresa; restano i checkpoint top-k delle epoche finite. Se un job non
  risponde c'è «Termina forzatamente».
- **Chiusura dell'app con job attivi**: «Metti in pausa e chiudi» (i training salvano il checkpoint, gli altri job
  vengono interrotti) oppure «Lascia in background» (i job continuano senza interfaccia).
- **Riapertura**: l'app si ricollega ai job ancora in esecuzione (barre, curve e GPU riprendono), elenca quelli in
  pausa in «Da riprendere» e segnala quelli terminati in modo anomalo (crash, spegnimento). Anche questi si
  possono riprendere dall'ultimo **checkpoint automatico**, scritto ogni `train.checkpoint_every_min` minuti
  (15 per default; `ssl.checkpoint_every_min` per DINO): si perde al massimo quel lavoro.
- Le **campagne** si mettono in pausa allo stesso modo: il passo in corso salva il checkpoint e la campagna si
  ferma; alla ripresa salta i passi completati e riprende quello in pausa dal suo checkpoint.

Dalla riga di comando lo stesso meccanismo si usa con `rsseg train --resume RUN_DIR` (o `rsseg pretrain --resume`).
Alla ripresa si possono cambiare solo le impostazioni che non alterano il training (`data.num_workers`, `eval.*`,
`train.checkpoint_every_min`, …).

### Come funziona la ripresa esatta

Il checkpoint contiene tutte le variabili del modello (compreso lo stato dei generatori casuali di dropout e
stochastic depth, il teacher EMA e il centro di DINO), le variabili dell'ottimizzatore (momenti di AdamW, contatore
dei passi, buffer dell'accumulo del gradiente), gli accumulatori delle metriche a metà epoca, lo stato delle callback
(early stopping, top-k, MLflow) e la **posizione nel flusso dei dati**. L'ordine dei crop e la loro augmentation
dipendono solo dal seed e dalla posizione (`StreamSampler`), non dal numero di worker né dalla storia del processo:
dopo la ripresa il training vede esattamente i batch che avrebbe visto senza interruzioni.

## 6. Monitoraggio della GPU

- **Rilevamento**: NVML (`nvidia-ml-py`) oppure `nvidia-smi`; nome, driver e versione CUDA supportata.
- **PyTorch vede la GPU?** Un controllo in un processo separato: con un PyTorch installato senza CUDA il training
  girerebbe sulla CPU senza errori. L'indicatore diventa giallo con il comando per reinstallare PyTorch con CUDA.
- **In tempo reale** (ogni secondo): utilizzo, memoria usata/totale, temperatura, potenza, frequenza, ventola,
  processi con la loro memoria (i job dell'app sono evidenziati), grafici degli ultimi 10 minuti, avvisi (memoria
  oltre il 95 %, temperatura oltre 85 °C).
- **Prima di avviare** un job sulla GPU l'app controlla che la GPU sia pronta, che la memoria libera basti per il
  picco misurato con il benchmark e che altri processi non occupino VRAM. I job GPU girano uno alla volta.
- **Storico**: durante ogni job i campioni vanno in `gpu_log.csv`, copiato nella cartella del run e visibile in
  «Risultati». Il training registra anche su MLflow, per epoca, utilizzo medio, picco di memoria della scheda e
  temperatura massima.

## 7. Inferenza

1. **Modello**: i run di `runs/` con i pesi; il suggerimento riassume encoder, pre-training, canali, epoche e mIoU.
   «Pesi» sceglie il checkpoint migliore o una delle epoche top-k.
2. **Immagini**: le tile di uno split dei dati del run (con ground truth: metriche e mappa degli errori) oppure
   immagini e cartelle qualsiasi. Un modello RGB + nDSM richiede un nDSM per immagine.
3. **Parametri**: finestra, passo, batch e test-time augmentation (default dalla configurazione del run).

Il job lancia `rsseg predict`, che scrive per ogni immagine la maschera delle classi, la versione colorata, la
mappa di confidenza e, con ground truth, gli errori, più `predictions.json`. Il visualizzatore mostra immagine,
predizione, ground truth o confidenza ed errori **con zoom e spostamento sincronizzati** (rotella e trascinamento),
oppure la predizione sovrapposta con opacità regolabile. La legenda accende e spegne le classi; passando il mouse
si leggono classe predetta, confidenza e classe vera del pixel. «Confronta con…» affianca un'altra inferenza sulla
stessa immagine con la mappa delle differenze (verde: ha ragione solo il secondo modello; rosso: solo il primo).
«Esporta pannello» salva immagine | predizione | verità | errori a piena risoluzione.

```bash
# lo stesso da riga di comando
rsseg predict --run runs/seg-maresunet_r18_imagenet-… --split test --out runs/predictions/r18-test
rsseg predict --run runs/seg-… immagini/ --out runs/predictions/nuove --tta
```

## 8. Tema scuro e tema chiaro

L'app parte in **tema scuro**. Il pulsante «☀ Tema chiaro» / «☾ Tema scuro» in basso a destra, nella barra di
stato, cambia tema al volo, grafici compresi; la scelta viene ricordata alla prossima apertura.

Lo stile è lo stesso dell'app video-downloader (`rsseg/gui/theme.py` ne è l'adattamento):

- **colori per significato**: ogni colore ha un ruolo (`accent`, `success`, `warning`, `danger`, testo attenuato,
  superfici); gli stati dei job usano sempre gli stessi toni: in corso indaco, in pausa giallo, completato verde,
  fallito o interrotto rosso, in coda o fermato grigio;
- **leggibilità verificata**: un test controlla che ogni testo abbia contrasto almeno 4,5:1 (WCAG AA) sulla
  superficie su cui compare, in entrambi i temi, comprese le percentuali stampate sulle barre di avanzamento;
- **grafici**: sfondo, assi e griglia seguono il tema; le serie usano otto colori in ordine fisso, validati per
  il daltonismo su entrambi gli sfondi; la stessa grandezza ha lo stesso colore in tutti i grafici
  (training blu, validazione arancione);
- **pulsanti per ruolo**: un solo pulsante principale per pagina (indaco pieno), le azioni distruttive in rosso,
  le icone (play, pausa, stop, cartella) disegnate nel codice e ricolorate con il tema.

Non cambiano con il tema i **colori delle classi** ISPRS (sono dati, identici alle figure della tesi) né le
**esportazioni** per i documenti: «Salva grafico PNG» (Risultati) salva sempre il grafico in tema chiaro,
«Esporta pannello…» (Inferenza), Markdown e CSV non dipendono dal tema. «Salva vista PNG» invece salva la vista
così come appare sullo schermo.

## 9. File dell'applicazione

| File | Contenuto |
|---|---|
| `.gui/jobs.json` | Elenco dei job (comando, stato, PID, cartella del run, stima) |
| `.gui/jobs/<id>/progress.json` | Avanzamento scritto dal comando (`RSSEG_PROGRESS_FILE`) |
| `.gui/jobs/<id>/control.json` | Richieste di pausa o stop (`RSSEG_CONTROL_FILE`) |
| `.gui/jobs/<id>/log.txt`, `stdout.json` | Log e risultato JSON del comando |
| `.gui/jobs/<id>/gpu_log.csv` | Campioni della GPU durante il job |
| `.gui/benchmarks.json` | Picchi di VRAM misurati, per configurazione |
| impostazioni utente Qt (`rsseg/rsseg-gui`) | Tema scelto (`dark` o `light`) |
| `<run>/resume/state.json`, `state.npz` | Checkpoint di ripresa (cancellato quando il run finisce) |

## 10. Codice

| Modulo | Ruolo |
|---|---|
| `rsseg/control.py` | Richieste di pausa/stop tramite file, codici di uscita 10 (pausa) e 11 (stop) |
| `rsseg/progress.py` | Avanzamento condiviso: stato, tempi, fasi, contatori, storico per epoca |
| `rsseg/runstate.py` | Checkpoint di ripresa visti dall'esterno, ambiente, override ammessi |
| `rsseg/train/resume.py` | `ResumeCheckpoint`, `fit_resumable`: salvataggio e ripresa esatta |
| `rsseg/eval/predict.py` | `rsseg predict` |
| `rsseg/gui/core/` | Parti senza Qt, testate senza display: job (`jobs.py`), tempi (`timing.py`), GPU (`gpu.py`), suggerimenti (`help.py`), run (`runs.py`) |
| `rsseg/gui/theme.py` | Palette scura e chiara, foglio di stile, icone disegnate, colori dei grafici |
| `rsseg/gui/context.py`, `widgets.py`, `actions.py`, `app.py`, `pages/` | Interfaccia Qt |

L'interfaccia non importa mai torch né keras (un test lo verifica): i comandi pesanti girano in processi separati,
così un errore di memoria non chiude l'app e la VRAM si libera del tutto alla fine di ogni job.
