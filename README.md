![GitHub repo size](https://img.shields.io/github/repo-size/mirkopat/tesi_rinocitologia)
![GitHub last commit](https://img.shields.io/github/last-commit/mirkopat/tesi_rinocitologia)
![GitHub language count](https://img.shields.io/github/languages/count/mirkopat/tesi_rinocitologia)

# Oltre il mAP: metriche di validazione cito-cliniche per il rilevamento automatico di citotipi nasali

**Tesi di Laurea in Informatica**

**Autore:** Mirko Patruno  
**Correlatrice:** Prof.ssa Nunzia Lomonte  
**Anno Accademico:** 2025/2026  

---

## 📋 Descrizione del Progetto

La citologia nasale è una tecnica diagnostica minimamente invasiva che permette di identificare i diversi endotipi di rinite attraverso l'analisi microscopica delle cellule nasali. L'interpretazione del referto segue protocolli standardizzati che includono il conteggio di 50 campi microscopici per vetrino e l'applicazione di griglie semi-quantitative (AICNA) per la classificazione degli endotipi (NARES, NARMA, NARESMA, NARNE).

Questa tesi si pone l'obiettivo di **confrontare metriche di object detection standard (mAP)** con **metriche cito-cliniche** per valutare modelli di deep learning sul dataset NMCD (Nasal Mucosa Cell Dataset). L'ipotesi è che il ranking dei modelli possa cambiare significativamente quando si passa da metriche di detection a metriche cliniche, evidenziando l'importanza di metriche specifiche per il dominio.

---

## 🎯 Obiettivi della Tesi

1. **Addestrare e confrontare tre architetture di object detection** diverse da quelle già valutate su NMCD:
   - YOLOv10 (nano e medium)
   - RF-DETR (transformer-based, 20/50/100 epoche)
   - D-FINE (transformer-based, 50 epoche)

2. **Derivare dalla ground truth per-istanza una ground truth clinica**:
   - A livello di campo microscopico
   - A livello di "paziente virtuale" (50 campi)

3. **Definire e applicare una batteria di metriche cito-cliniche**:
   - Accuratezza dell'endotipo predetto
   - Sensibilità sulle classi rare (mastociti, linfociti)
   - Corrispondenza dei gradi semi-quantitativi AICNA

4. **Verificare se il ranking dei modelli cambia** tra metriche di detection e metriche cliniche

---

## 📁 Struttura del Repository
```bash
tesi_rinocitologia/
│
├── data/
│ └── NMCD.coco/                       # Dataset in formato COCO
│ ├── train/                           # 400 immagini per l'addestramento
│ ├── valid/                           # 50 immagini per la validazione
│ ├── test/                            # 50 immagini per il test
│ └── data.yaml                        # Configurazione per YOLO
│
├── notebooks/
│ ├── 01_esplorazione_dataset.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_metriche_cliniche.ipynb
│
├── src/
│ ├── convert_coco_to_yolo.py          # Conversione COCO → YOLO
│ ├── train_yolo.py                    # Addestramento YOLOv10n (50 ep)
│ ├── train_yolo_medium.py             # Addestramento YOLOv10m (50 ep)
│ ├── train_yolo_nano_v2.py            # Addestramento YOLOv10n (100 ep)
│ ├── train_rfdetr.py                  # Addestramento RF-DETR
│ ├── evaluate_clinical.py             # Valutazione clinica YOLO
│ ├── evaluate_rfdetr_clinical.py      # Valutazione clinica RF-DETR
│ ├── calculate_ap_per_class_rfdetr.py # AP per classe RF-DETR
│ ├── metrics_cliniche.py              # Metriche cliniche
│ └── utils.py                         # Funzioni di utilità
│
├── dfine_scripts/                     # Script personalizzati per D-FINE
│ ├── README.md
│ ├── evaluate_clinical_dfine.py
│ ├── calculate_ap_per_class.py
│ ├── convert_annotations.py
│ ├── fix_categories.py
│ ├── fix_filenames.py
│ └── configs/
│ ├── custom_detection.yml
│ ├── custom_optimizer.yml
│ ├── custom_runtime.yml
│ └── dfine_hgnetv2_s_custom.yml
│
├── test/                              # Script di test
├── requirements.txt                   # Dipendenze Python
└── README.md                          # Questo file
```

**Note:**
- Le cartelle `results/`, `runs/`, `weights/` e i file `*.pt`, `*.pth`, `*.safetensors` **non sono incluse** nel repository perché troppo pesanti. Chi vuole riprodurre i risultati deve riaddestrare i modelli.
- La cartella `D-FINE/` **non è inclusa** perché è un repository esterno. Vedi `dfine_scripts/README.md` per le istruzioni.
- Le immagini del dataset (`data/NMCD.coco/*/images/`) **non sono incluse** perché pesanti. Il dataset va scaricato separatamente.

---

## 🔧 Setup e Installazione

### Prerequisiti

- **Python 3.10.4**
- **CUDA 12.6** (driver 560.94) - consigliato per GPU NVIDIA
- **GPU consigliata**: NVIDIA GeForce GTX 1060 6GB (o superiore)

### Installazione

```bash
# 1. Clona il repository
git clone https://github.com/mirkopat/tesi_rinocitologia.git
cd tesi_rinocitologia

# 2. Crea un ambiente virtuale (opzionale ma consigliato)
python -m venv venv
source venv/bin/activate  # Su Windows: .\venv\Scripts\activate

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Installa PyTorch con CUDA (se hai GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 5. Per CPU (senza CUDA)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

```

## 🚀 Addestramento dei Modelli
### YOLOv10
```bash
# YOLOv10n - 50 epoche (modello base)
python src/train_yolo.py

# YOLOv10m - 50 epoche (modello medium)
python src/train_yolo_medium.py

# YOLOv10n - 100 epoche (modello con più epoche)
python src/train_yolo_nano_v2.py
```

### RF-DETR
```bash
# RF-DETR - 20 epoche
python src/train_rfdetr.py  # Modifica EPOCHS = 20

# RF-DETR - 50 epoche
python src/train_rfdetr.py  # Modifica EPOCHS = 50

# RF-DETR - 100 epoche
python src/train_rfdetr.py  # Modifica EPOCHS = 100
```

### D-FINE

Vedi le istruzioni dettagliate in `dfine_scripts/README.md`.


## 📊 Valutazione
### Calcolo delle metriche cliniche:
```bash
# Per YOLO
python src/evaluate_clinical.py

# Per RF-DETR
python src/evaluate_rfdetr_clinical.py

# Per D-FINE
cd D-FINE
python evaluate_clinical_dfine.py
```

### Output atteso

```text
🔬 Paziente 1: 50 immagini
   GT: {'epithelial': 503, 'goblet_cell': 51, ...}
   Pred: {'epithelial': 450, 'eosinophil': 55, ...}

==================================================
CONFRONTO ENDOTIPI
==================================================
Paziente 1: GT=nares           | Pred=nares           | ✅

📊 Accuratezza endotipi: 1/1 = 100.00%
```

---

## 📊 Risultati

### Metriche di Detection

| Modello | Epoche | mAP50 | mAP50-95 | Precision | Recall | F1 | mast_cell AP | eosinophil AP | neutrophil AP | Tempo |
|---------|--------|-------|----------|-----------|--------|-----|--------------|---------------|---------------|-------|
| YOLOv10n | 50 | 0.341 | 0.180 | 0.714 | 0.312 | - | 0.762 | 0.623 | 0.018 | ~1 ora |
| YOLOv10n | 100 | 0.414 | 0.245 | 0.648 | 0.357 | - | 0.708 | 0.667 | 0.049 | ~1.5 ore |
| YOLOv10m | 50 | **0.498** | **0.256** | 0.545 | 0.479 | - | **0.823** | **0.786** | **0.426** | ~2 ore |
| RF-DETR | 20 | 0.397 | 0.219 | 0.359 | 0.418 | 0.377 | 0.000 | 0.200 | 0.295 | ~44 min |
| RF-DETR | 50 | 0.461 | 0.251 | 0.520 | **0.599** | 0.463 | 0.000 | 0.289 | 0.318 | ~2.3 ore |
| RF-DETR | 100 | 0.430 | 0.236 | 0.345 | 0.469 | 0.385 | 0.000 | 0.269 | 0.323 | ~4.5 ore |
| D-FINE | 50 | 0.390 | 0.215 | - | - | - | 0.000 | 0.216 | 0.325 | ~3.3 ore |

### Metriche Cliniche (Paziente Virtuale)

| Modello | Epoche | mAP50 | Endotipo Predetto | Endotipo Ground Truth | Corretto? | Accuratezza Gradi AICNA |
|---------|--------|-------|-------------------|----------------------|-----------|-------------------------|
| YOLOv10n | 50 | 0.341 | NARESMA | NARES | ❌ | 25.00% (2/8) |
| YOLOv10n | 100 | 0.414 | NARESMA | NARES | ❌ | 25.00% (2/8) |
| YOLOv10m | 50 | **0.498** | NARESMA | NARES | ❌ | 37.50% (3/8) |
| RF-DETR | 20 | 0.397 | NARNE | NARES | ❌ | 62.50% (5/8) |
| RF-DETR | 50 | 0.461 | NARNE | NARES | ❌ | **75.00% (6/8)** |
| RF-DETR | 100 | 0.430 | NARNE | NARES | ❌ | 62.50% (5/8) |
| D-FINE | 50 | 0.390 | **NARES** | NARES | **✅** | 62.50% (5/8) |

> **Nota:** Solo D-FINE (con soglia 0.6) classifica correttamente l'endotipo NARES. I modelli YOLO sovrastimano i mastociti (predicono 502-564 vs 2 ground truth), portando a una falsa diagnosi di NARESMA. I modelli RF-DETR sottostimano eosinofili e mastociti, portando a una falsa diagnosi di NARNE.

### Analisi dei Gradi AICNA

| Citotipo | GT | YOLOv10n | YOLOv10m | RF-DETR 20 | RF-DETR 50 | RF-DETR 100 | D-FINE |
|----------|-----|----------|----------|------------|------------|-------------|--------|
| epithelial | ++++ | +/++ | ++ | ++++ | ++++ | ++++ | ++++ |
| neutrophil | ++++ | 0 | 0 | ++++ | ++++ | ++++ | ++++ |
| eosinophil | ++++ | ++++ | ++++ | +++ | ++++ | +++ | ++++ |
| mast_cell | + | ++++ | ++++ | 0 | 0 | 0 | 0 |
| lymphocyte | +++ | ++ | + | 0 | 0 | 0 | 0 |
| goblet_cell | + | 0 | 0 | + | + | + | + |
| metaplastic | + | 0 | 0 | + | + | + | + |
| ciliated | + | + | + | + | + | + | 0 |
| **Accuratezza** | - | **25%** | **37.5%** | **62.5%** | **75%** | **62.5%** | **62.5%** |

### 🔍 Conclusioni Chiave

1. **Il disallineamento è confermato:** I modelli con mAP più alta non sono necessariamente quelli che classificano correttamente l'endotipo.
2. **Solo D-FINE classifica correttamente NARES** (con soglia 0.6).
3. **I modelli YOLO sovrastimano i mastociti**, portando a una falsa diagnosi di NARESMA.
4. **I modelli RF-DETR sottostimano eosinofili e mastociti**, portando a una falsa diagnosi di NARNE.
5. **L'accuratezza dei gradi AICNA** è massima per RF-DETR (50 ep) con **75.00%**.

## 📚 Riferimenti Bibliografici
- **Camporeale et al. (2026)** - A nasal cytology dataset for object detection and deep learning
Biomedical Signal Processing and Control

- **Macchi et al. (2026)** - Standardization of Nasal Cytology: An Expert-based Delphi Consensus
Current Allergy and Asthma Reports

- **Gelardi (2025)** - Nasal cytology in the rhinology-allergy clinic: From rhinitis to chronic rhinosinusitis with nasal polyps
Asia Pacific Allergy

- **Gelardi (2026)** - Nasal Cytology as a Cellular Window into Epithelial Dysfunction and Type 2 Inflammation
Cells

- **Shrikrishna & Deepa (2025)** - The Application and Diagnostic Accuracy of Artificial Intelligence in Rhinology
Cureus

- **Zhang et al. (2026)** - Development of Artificial Intelligence for Quantitative Assessment of Nasal Inflammatory Cytology
International Forum of Allergy & Rhinology

## 📝 Note Tecniche
### Su Windows
- Usare `workers=0` per evitare deadlock nel multiprocessing

- Aggiungere `if __name__ == '__main__':` negli script

- Usare virgolette per percorsi con spazi

### Sul Dataset
- I nomi delle classi con spazi (`mast cell`, `epithelial ciliated`) sono stati rinominati per YOLO (`mast_cell`, `ciliated`)

- La classe `cells` (ID 0) è stata esclusa perché generica

- `emazia` (eritrociti) e `artefatto` sono esclusi dalle metriche cliniche

### Sui Modelli
- **YOLOv10**: Addestrato con Ultralytics, 3 configurazioni testate

- **RF-DETR**: Addestrato con la libreria rfdetr, 3 configurazioni testate

- **D-FINE**: Addestrato clonando il repository ufficiale, 1 configurazione testata

- **RT-DETR**: Testato ma abbandonato per bug di training (0 predizioni)

- **EfficientDet**: Testato ma abbandonato per conflitti di libreria

## 📄 Licenza

Questo progetto è realizzato a scopo di tesi di laurea. Tutti i diritti riservati.

---

**Ultimo aggiornamento:** Settembre 2026