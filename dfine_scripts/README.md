# Script personalizzati per D-FINE

Questi script sono stati creati per addestrare e valutare D-FINE sul dataset NMCD.

---

## Prerequisiti

1. Clonare il repository ufficiale di D-FINE:
   ```bash
   git clone https://github.com/Peterande/D-FINE.git
   ```

2. Installare le dipendenze:
   ```bash
   pip install -r D-FINE/requirements.txt
   ```

---

## Struttura

- `evaluate_clinical_dfine.py`: Valutazione clinica di D-FINE (endotipo, gradi AICNA)
- `calculate_ap_per_class.py`: Calcolo AP per classe con pycocotools
- `calculate_ap_per_class_rfdetr.py`: Calcolo AP per classe per RF-DETR
- `convert_annotations.py`: Conversione annotazioni COCO per D-FINE
- `fix_categories.py`: Correzione nomi categorie
- `fix_filenames.py`: Correzione percorsi immagini
- `configs/`: Configurazioni YAML per D-FINE

---

## Come Eseguire

### 1. Clonare D-FINE nella root del progetto

```bash
git clone https://github.com/Peterande/D-FINE.git
```

### 2. Copiare gli script e le configurazioni nelle rispettive cartelle

```bash
copy dfine_scripts\*.py D-FINE\
copy dfine_scripts\configs\*.yml D-FINE\configs\dfine\custom\
```

### 3. Addestrare D-FINE

```bash
cd D-FINE
python train.py -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml
```

### 4. Valutare D-FINE

```bash
python evaluate_clinical_dfine.py
python calculate_ap_per_class.py
```

---

## Risultati Ottenuti

| Metrica | Valore |
|---------|--------|
| mAP50 | 0.390 |
| mAP50-95 | 0.215 |
| Endotipo predetto | NARNE (errato, ground truth: NARES) |
| mast_cell AP | 0.000 |
| eosinophil AP | 0.216 |
| neutrophil AP | 0.325 |