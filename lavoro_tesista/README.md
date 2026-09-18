# Classificazione dei citotipi nasali - codice tesi

Questo repository contiene il codice usato per gli esperimenti della tesi:

```text
Metodi spiegabili per la classificazione dei citotipi nasali:
confronto tra feature manuali, CNN e transfer learning
```

L'obiettivo e classificare crop cellulari estratti dal dataset NMCD
(`Nasal Mucosa Cell Dataset`) confrontando tre famiglie di approcci:

- feature manuali con modelli classici di machine learning;
- CNN residuale addestrata da zero;
- transfer learning con backbone pre-addestrati.

## Struttura

```text
.
|-- README.md
|-- requirements.txt
|-- src/
|   `-- nmcd_common.py
|-- scripts/
|   |-- 01_inspect_coco.py
|   |-- 02_export_cell_crops.py
|   |-- 03_extract_handcrafted_features.py
|   |-- 04_train_feature_baseline.py
|   |-- 05_review_errors.py
|   |-- 06_train_cnn_baseline.py
|   |-- 07_train_transfer_learning.py
|   `-- 08_build_project_report.py
|-- docs/
|   |-- CODE_MAP.md
|   |-- experiment_log.md
|   `-- aggiornamento_progetto.md
`-- configs/
    `-- README.md
```

Il dataset e gli output non sono inclusi nel pacchetto codice per evitare file
pesanti e dati generati. Gli script assumono questa struttura locale:

```text
Dataset Rinocitologia/NMCD.coco/
|-- train/
|-- valid/
`-- test/
```

In alternativa e possibile passare un percorso diverso con `--dataset-root`.

## Setup

Ambiente consigliato: Python 3.11 o 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Per gli esperimenti deep learning su CPU/GPU puo essere necessario installare
PyTorch seguendo le istruzioni ufficiali per la propria piattaforma.

## Esecuzione pipeline

1. Analisi del dataset COCO:

```powershell
python scripts/01_inspect_coco.py
```

2. Estrazione dei crop cellulari dalle annotazioni:

```powershell
python scripts/02_export_cell_crops.py --padding-ratio 0.08
```

3. Estrazione delle feature manuali:

```powershell
python scripts/03_extract_handcrafted_features.py
```

4. Addestramento delle baseline classiche:

```powershell
python scripts/04_train_feature_baseline.py
```

5. Analisi degli errori:

```powershell
python scripts/05_review_errors.py --split test
```

6. Addestramento della CNN residuale:

```powershell
python scripts/06_train_cnn_baseline.py --output-root outputs/cnn_residual
```

7. Transfer learning:

```powershell
python scripts/07_train_transfer_learning.py --backbone resnet18 --output-root outputs/transfer_resnet18
python scripts/07_train_transfer_learning.py --backbone efficientnet_b0 --output-root outputs/transfer_learning
```

8. Generazione del report progettuale opzionale:

```powershell
python scripts/08_build_project_report.py
```

## Output principali

Gli script scrivono i risultati in `outputs/`, cartella esclusa dal pacchetto:

- `outputs/reports/`: riepiloghi del dataset;
- `outputs/metadata/cell_crops.csv`: metadati dei crop estratti;
- `outputs/features/handcrafted_features.csv`: feature manuali;
- `outputs/models/`: baseline classiche e metriche;
- `outputs/cnn_residual/`: risultati CNN;
- `outputs/transfer_resnet18/`: risultati transfer learning.

## Note

- Gli split `train`, `valid` e `test` sono mantenuti separati come nel dataset.
- Le classi rare non vengono eliminate: lo sbilanciamento e gestito con pesi di
  classe e metriche macro.
- La metrica principale usata per selezionare i modelli e la macro-F1, perche
  descrive meglio il comportamento sulle classi poco rappresentate.
