# semantic-segmentation-potsdam
 
Per lanciare il training eseguire il seguente comando:
(prima di eseguire lo script assicurarsi che nel file di configurazione i percorsi del training set e del validation set siano stati correttamente inseriti)

```
yes | python config_changer.py --epochs=150  --model_name='maresunet18' --train_dataset_length=-1 --valid_dataset_length=-1 --vq_flag=False
cat config/train_pl_ss.yaml
python main_pl_ss.py
```

dove le seguenti argomenti di inputs rappresentano

- model_name : nome del modello selezionabile tra i seguenti elencati 
  - "maresunet18"
  - "maresunet18-vq"
  - "maresunet50"
  - "maresunet50-vq"
  - "maresunet50-pretrain-EsViT"
  - "maresunet50-vq-pretrain-EsViT"
- train_dataset_length : numero di path da usare per il training set (-1 se si vuole usare il training set per intero)
- valid_dataset_length : numero di path da usare per il validation set (-1 se si vuole usare il validation set per intero)
- vq_flag : impostare il valore a "True" se si fa uso di modello con implentazione di vector quantization "False" altrimenti
