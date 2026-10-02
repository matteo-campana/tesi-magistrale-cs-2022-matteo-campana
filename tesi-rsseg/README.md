# Tesi (nuova stesura, LaTeX)

Nuova stesura della tesi magistrale, basata sulla reimplementazione del repository (`src/rsseg/`, `docs/`).
Riprende classe, stile e palette della tesi del 2022 (`legacy/tesi-magistrale/Tesi_Matteo_Campana_1692599_V4.pdf`).

## Compilazione

Serve una distribuzione TeX Live con la classe `sapthesis` (pacchetto `texlive-publishers` su Debian/Ubuntu).

```bash
cd tesi-rsseg
latexmk          # pdflatex + bibtex; il PDF finisce in build/main.pdf
latexmk -c       # pulizia dei file intermedi
```

In `main.tex`, `\bozzatrue` mostra le note `\todo`, i segnaposto e l'elenco delle parti da completare.
`\bozzafalse` produce la versione da consegnare: le note spariscono.

## Struttura

```
main.tex                    frontespizio, ordine dei capitoli, bibliografia
impostazioni/
  colori.tex                palette (rosso Sapienza, colori dei link, classi ISPRS, ruoli negli schemi)
  pacchetti.tex             pacchetti e impostazioni (babel, hyperref, todonotes, TikZ, ...)
  comandi.tex               enunciati, notazione matematica, segnaposto (\Segnaposto, \ND, \FiguraMancante)
capitoli/
  abstract.tex, ringraziamenti.tex
  01_introduzione.tex       contesto, difficoltà, obiettivi, domande di ricerca RQ1-RQ6, contributi
  02_premesse.tex           stato dell'arte, tesi del 2022, lettura critica, correzioni adottate
  03_teoria.tex             formulazione, LAM, VQ, DINO, nDSM, metriche, finestra scorrevole
  04_metodologia.tex        ipotesi, dati, split, addestramento, implementazione, budget, statistica
  05_esperimenti.tex        tabelle e figure per RQ, con valori n.d. e note \todo (da completare)
  06_prospettive.tex        stato del lavoro, attività rimanenti, prospettive, struttura delle conclusioni
figure/                     schemi TikZ e grafici pgfplots (inclusi con \input)
bibliografia.bib            riferimenti (BibTeX)
sapthesis-it.bst            copia di sapthesis.bst con le parole fisse in italiano (stessa impaginazione)
```

## Stile ripreso dalla tesi del 2022

| Elemento | Scelta |
|---|---|
| Classe | `sapthesis`, A4 fronte-retro, Latin Modern, testatine con filetto |
| Colori | rosso Sapienza `RGB(130,36,51)`; cornici dei link di hyperref rosse (riferimenti interni), verdi (citazioni), ciano (URL) |
| Paragrafi | separati da uno spazio verticale, senza rientro |
| Tabelle | griglia completa, intestazioni in grassetto, didascalia sopra |
| Bibliografia | numerica, ordinata per autore, nomi in maiuscoletto (`sapthesis.bst`, qui localizzato in `sapthesis-it.bst`) |

## Completare gli esperimenti

I valori segnati come `\ND` nel capitolo 5 vanno sostituiti con media ± deviazione standard sui tre seed, presi dai
report delle campagne (`rsseg campaign …`, `runs/campaigns/<campagna>/report.md`) e da MLflow. Le figure
`\FiguraMancante{…}` descrivono il contenuto atteso. Il capitolo 6 va sostituito dalle conclusioni solo a campagne
concluse.
