# src/calculate_ap_per_class_yolo.py
# Calcola AP per classe per YOLO usando pycocotools

import json
import os
from pathlib import Path
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# CONFIGURAZIONE
MODEL_PATH = "runs/detect/results/yolo11/exp/weights/best.pt"
DATA_DIR = "data/NMCD.coco"
SPLIT = "valid"
CONFIDENCE_THRESHOLD = 0.25

# Carica modello
model = YOLO(MODEL_PATH)

# Carica annotazioni
ann_file = Path(DATA_DIR) / SPLIT / "_annotations.coco.json"
with open(ann_file, 'r') as f:
    coco_data = json.load(f)

with open('temp_gt.json', 'w') as f:
    json.dump(coco_data, f)

# Genera predizioni
predictions = []
for img in tqdm(coco_data['images'], desc="Inferenza"):
    img_path = Path(DATA_DIR) / SPLIT / "images" / Path(img['file_name']).name
    if not img_path.exists():
        continue
    
    results = model(img_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            predictions.append({
                'image_id': img['id'],
                'category_id': int(box.cls[0]) + 1,  # +1 perché COCO parte da 1
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'score': float(box.conf[0])
            })

with open('temp_dt.json', 'w') as f:
    json.dump(predictions, f)

# Valuta con COCOeval
coco_gt = COCO('temp_gt.json')
coco_dt = coco_gt.loadRes('temp_dt.json')
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# AP per classe
print("\nAP per classe (IoU=0.50):")
category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
for i, cat_id in enumerate(coco_gt.getCatIds()):
    ap = coco_eval.eval['precision'][:, :, i, 0, 2]
    ap_value = ap[ap > -1].mean() if len(ap[ap > -1]) > 0 else 0.0
    print(f"  {category_names[cat_id]}: {ap_value:.3f}")

os.remove('temp_gt.json')
os.remove('temp_dt.json')