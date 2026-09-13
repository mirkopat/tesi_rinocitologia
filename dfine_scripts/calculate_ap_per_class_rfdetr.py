# src/calculate_ap_per_class_rfdetr.py
#
# SCOPO: Calcolare AP per classe per RF-DETR usando pycocotools.
# AUTORE: Mirko Patruno
# DATA: Settembre 2026

import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os

from rfdetr import RFDETRBase
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ============================================================================
# 1. CONFIGURAZIONE
# ============================================================================

MODEL_PATH = "results/rfdetr"  # "results/rfdetr" per 20 epoche, "results/rfdetr_50ep" per 50 epoche e "results/rfdetr_100ep" per 100 epoche
DATA_DIR = "data/NMCD.coco"
CONFIDENCE_THRESHOLD = 0.6

# ============================================================================
# 2. CARICA MODELLO
# ============================================================================

print(f"🔍 Caricamento RF-DETR da: {MODEL_PATH}")
model = RFDETRBase(pretrain_weights=f"{MODEL_PATH}/checkpoint_best_ema.pth")

# ============================================================================
# 3. CARICA ANNOTAZIONI
# ============================================================================

ann_file = Path(DATA_DIR) / "valid" / "_annotations.coco.json"
with open(ann_file, 'r') as f:
    coco_data = json.load(f)

# Salva il ground truth in un file temporaneo per COCO
with open('temp_gt_rfdetr.json', 'w') as f:
    json.dump(coco_data, f)

# ============================================================================
# 4. GENERA PREDIZIONI IN FORMATO COCO
# ============================================================================

predictions = []

# Mappa dei nomi delle classi: COCO originale -> indice 0-based (per RF-DETR)
# Nota: RF-DETR usa i nomi originali del dataset COCO
category_names = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id']) if cat['name'] != 'cells']
# Crea una mappa: nome -> id COCO (per il file di output)
name_to_coco_id = {cat['name']: cat['id'] for cat in coco_data['categories']}

print(f"\n📊 Inferenza su {len(coco_data['images'])} immagini...")

for img in tqdm(coco_data['images'], desc="Inferenza"):
    img_path = Path(DATA_DIR) / "valid" / "images" / Path(img['file_name']).name
    if not img_path.exists():
        continue
    
    image = Image.open(img_path).convert('RGB')
    
    # Inferenza con RF-DETR
    detections = model.predict(image, threshold=CONFIDENCE_THRESHOLD)
    
    # Accedi agli attributi dell'oggetto Detections
    if hasattr(detections, 'class_id') and hasattr(detections, 'confidence'):
        # Ottieni i nomi delle classi dal dataset 'data'
        if hasattr(detections, 'data') and 'class_name' in detections.data:
            class_names = detections.data['class_name']
        else:
            class_names = None
        
        for idx, (class_id, confidence, box) in enumerate(zip(
            detections.class_id, detections.confidence, detections.xyxy
        )):
            if confidence > CONFIDENCE_THRESHOLD:
                # Ottieni il nome della classe
                if class_names is not None and idx < len(class_names):
                    cat_name = class_names[idx]
                else:
                    cat_name = None
                
                # Salta 'cells' se presente
                if cat_name == 'cells':
                    continue
                
                # Ottieni l'ID COCO corrispondente
                if cat_name in name_to_coco_id:
                    coco_cat_id = name_to_coco_id[cat_name]
                else:
                    continue
                
                # Converti bbox in formato COCO [x, y, w, h]
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                
                predictions.append({
                    'image_id': img['id'],
                    'category_id': coco_cat_id,
                    'bbox': [x1, y1, w, h],
                    'score': float(confidence.item())
                })

print(f"\n📊 Predizioni totali: {len(predictions)}")

with open('temp_dt_rfdetr.json', 'w') as f:
    json.dump(predictions, f)

# ============================================================================
# 5. VALUTA CON COCOEVAL
# ============================================================================

coco_gt = COCO('temp_gt_rfdetr.json')
coco_dt = coco_gt.loadRes('temp_dt_rfdetr.json')
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# ============================================================================
# 6. ESTRAI AP PER CLASSE
# ============================================================================

print("\n📊 AP per classe (IoU=0.50):")
category_names_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

for i, cat_id in enumerate(coco_gt.getCatIds()):
    cat_name = category_names_map.get(cat_id, f"Unknown({cat_id})")
    ap = coco_eval.eval['precision'][:, :, i, 0, 2]  # AP@0.5
    ap_value = ap[ap > -1].mean() if len(ap[ap > -1]) > 0 else 0.0
    print(f"  {cat_name}: {ap_value:.3f}")

# Pulizia
os.remove('temp_gt_rfdetr.json')
os.remove('temp_dt_rfdetr.json')

print("\n✅ Valutazione completata!")