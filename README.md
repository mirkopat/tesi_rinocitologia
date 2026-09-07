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
   - RT-DETR (transformer-based)
   - EfficientDet (compound scaling)

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

```
tesi_rinocitologia/
│
├── data/
│   └── NMCD.coco/              # Dataset in formato COCO
│       ├── train/              # 400 immagini per l'addestramento
│       ├── valid/              # 50 immagini per la validazione
│       ├── test/               # 50 immagini per il test
│       └── data.yaml           # Configurazione per YOLO
│
├── notebooks/
│   ├── 01_esplorazione_dataset.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_metriche_cliniche.ipynb
│
├── src/
│   ├── convert_coco_to_yolo.py   # Conversione COCO → YOLO
│   ├── train_yolo.py             # Addestramento YOLOv10n (50 ep)
│   ├── train_yolo_medium.py      # Addestramento YOLOv10m (50 ep)
│   ├── train_yolo_nano_v2.py     # Addestramento YOLOv10n (100 ep)
│   ├── train_rtdetr.py           # Addestramento RT-DETR
│   ├── train_efficientdet.py     # Addestramento EfficientDet
│   ├── metrics_cliniche.py       # Metriche cliniche
│   ├── evaluate_clinical.py      # Valutazione clinica dei modelli
│   └── utils.py                  # Funzioni di utilità
│
├── results/                     # Output dei modelli
│   ├── yolo10/                  # YOLOv10n (50 ep)
│   ├── yolo10_medium/           # YOLOv10m (50 ep)
│   ├── yolo10_nano_v2/          # YOLOv10n (100 ep)
│   ├── rtdetr/                  # RT-DETR (da fare)
│   └── efficientdet/            # EfficientDet (da fare)
│
├── requirements.txt             # Dipendenze Python
└── README.md                    # Questo file
```

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

---

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

### RT-DETR (da sviluppare)

```bash
python src/train_rtdetr.py
```

### EfficientDet (da sviluppare)

```bash
python src/train_efficientdet.py
```

---

## 📊 Valutazione

### Calcolo delle metriche cliniche

```bash
python src/evaluate_clinical.py
```

### Output atteso

```
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

## 📊 Risultati Preliminari

### Metriche di Detection (mAP50)

| Modello | Epoche | mAP50 | mast_cell AP | eosinophil AP | neutrophil AP |
|---------|--------|-------|--------------|---------------|---------------|
| YOLOv10n | 50 | 0.341 | 0.762 | 0.623 | 0.018 |
| YOLOv10n | 100 | 0.414 | 0.708 | 0.667 | 0.049 |
| YOLOv10m | 50 | 0.498 | 0.823 | 0.786 | 0.426 |
| RT-DETR | TBD | TBD | TBD | TBD | TBD |
| EfficientDet | TBD | TBD | TBD | TBD | TBD |

### Metriche Cliniche

| Modello | Endotype Accuracy |
|---------|-------------------|
| YOLOv10n (50 ep) | 0% |
| YOLOv10n (100 ep) | 0% |
| YOLOv10m (50 ep) | 0% |

> **Nota**: I risultati clinici sono in fase di ottimizzazione. Il modello sta attualmente sovrastimando i mastociti a causa dello sbilanciamento delle classi.

---

## 📚 Riferimenti Bibliografici

1. **Camporeale et al. (2026)** - *A nasal cytology dataset for object detection and deep learning*  
   Biomedical Signal Processing and Control

2. **Macchi et al. (2026)** - *Standardization of Nasal Cytology: An Expert-based Delphi Consensus*  
   Current Allergy and Asthma Reports

3. **Gelardi (2025)** - *Nasal cytology in the rhinology-allergy clinic: From rhinitis to chronic rhinosinusitis with nasal polyps*  
   Asia Pacific Allergy

4. **Gelardi (2026)** - *Nasal Cytology as a Cellular Window into Epithelial Dysfunction and Type 2 Inflammation*  
   Cells

5. **Shrikrishna & Deepa (2025)** - *The Application and Diagnostic Accuracy of Artificial Intelligence in Rhinology*  
   Cureus

6. **Zhang et al. (2026)** - *Development of Artificial Intelligence for Quantitative Assessment of Nasal Inflammatory Cytology*  
   International Forum of Allergy & Rhinology

---

## 📝 Note Tecniche

### Su Windows
- Usare `workers=0` per evitare deadlock nel multiprocessing
- Aggiungere `if __name__ == '__main__':` negli script
- Usare virgolette per percorsi con spazi

### Sul Dataset
- I nomi delle classi con spazi (`mast cell`, `epithelial ciliated`) sono stati rinominati per YOLO (`mast_cell`, `ciliated`)
- La classe `cells` (ID 0) è stata esclusa perché generica
- `emazia` (eritrociti) e `artefatto` sono esclusi dalle metriche cliniche

## 📄 Licenza

Questo progetto è realizzato a scopo di tesi di laurea. Tutti i diritti riservati.

---

**Ultimo aggiornamento:** Settembre 2026