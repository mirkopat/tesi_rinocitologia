# src/evaluate_rfdetr_clinical.py
# Valutazione clinica di RF-DETR sul dataset NMCD

from rfdetr import RFDETRBase
import json
from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm import tqdm

# ============================================================================
# 1. CONFIGURAZIONE
# ============================================================================

MODEL_PATH = "results/rfdetr_100ep"
DATA_DIR = "data/NMCD.coco"
CONFIDENCE_THRESHOLD = 0.25

# Mappa dei nomi: COCO originale -> nome YOLO/clinico
CLASS_NAME_MAP = {
    'epithelial': 'epithelial',
    'neutrophil': 'neutrophil',
    'eosinophil': 'eosinophil',
    'mast cell': 'mast_cell',
    'lymphocyte': 'lymphocyte',
    'muciparous': 'goblet_cell',
    'metaplastic': 'metaplastic',
    'epithelial ciliated': 'ciliated',
    'emazia': 'erythrocyte',
    'artefatto': 'artifact'
}

# Classi cliniche (escluse emazia e artefatto)
CLINICAL_CLASSES = ['epithelial', 'neutrophil', 'eosinophil', 'mast_cell', 
                    'lymphocyte', 'goblet_cell', 'metaplastic', 'ciliated']

print(f"🔍 Caricamento RF-DETR da: {MODEL_PATH}")

model = RFDETRBase(pretrain_weights=f"{MODEL_PATH}/checkpoint_best_ema.pth")

# ============================================================================
# 2. CARICA ANNOTAZIONI
# ============================================================================

ann_file = Path(DATA_DIR) / "valid" / "_annotations.coco.json"
with open(ann_file, 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']
categories = {cat['id']: cat['name'] for cat in coco_data['categories'] if cat['id'] != 0}

print(f"   Immagini: {len(images)}")
print(f"   Annotazioni: {len(annotations)}")
print(f"   Classi: {len(categories)}")

# Crea un indice delle annotazioni per image_id
anns_by_image = {}
for ann in annotations:
    img_id = ann['image_id']
    if img_id not in anns_by_image:
        anns_by_image[img_id] = []
    anns_by_image[img_id].append(ann)

# ============================================================================
# 3. PAZIENTE VIRTUALE (50 CAMPI)
# ============================================================================

gt_counts = Counter()
pred_counts = Counter()

print("\n📊 Valutazione su 50 immagini (paziente virtuale)...")

for img in tqdm(images, desc="Inferenza"):
    img_path = Path(DATA_DIR) / "valid" / "images" / Path(img['file_name']).name
    
    if not img_path.exists():
        continue
    
    # Ground truth (mappa i nomi)
    for ann in anns_by_image.get(img['id'], []):
        cat_id = ann['category_id']
        if cat_id != 0 and cat_id in categories:
            cat_name_orig = categories[cat_id]
            cat_name = CLASS_NAME_MAP.get(cat_name_orig, cat_name_orig)
            if cat_name in CLINICAL_CLASSES:
                gt_counts[cat_name] += 1
    
    # Predizione
    image = Image.open(img_path).convert('RGB')
    detections = model.predict(image, threshold=CONFIDENCE_THRESHOLD)
    
    if hasattr(detections, 'class_id') and hasattr(detections, 'confidence'):
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            if confidence > CONFIDENCE_THRESHOLD:
                # Usa il nome della classe dal dataset 'data' di RF-DETR
                if hasattr(detections, 'data') and 'class_name' in detections.data:
                    class_names = detections.data['class_name']
                    idx = list(detections.class_id).index(class_id)
                    if idx < len(class_names):
                        cat_name_orig = class_names[idx]
                        cat_name = CLASS_NAME_MAP.get(cat_name_orig, cat_name_orig)
                    else:
                        cat_name = 'unknown'
                else:
                    cat_name = 'unknown'
                
                if cat_name in CLINICAL_CLASSES:
                    pred_counts[cat_name] += 1

# ============================================================================
# 4. RISULTATI
# ============================================================================

print("\n📊 Risultati (solo classi cliniche):")
print(f"   Ground truth totali: {sum(gt_counts.values())}")
print(f"   Predizioni totali: {sum(pred_counts.values())}")
print(f"   Recall approssimativo: {sum(pred_counts.values())/sum(gt_counts.values()):.2%}")

print("\n   Distribuzione per classe clinica:")
for name in CLINICAL_CLASSES:
    gt = gt_counts.get(name, 0)
    pred = pred_counts.get(name, 0)
    if gt > 0 or pred > 0:
        print(f"     {name}: pred={pred}, gt={gt}")

print("\n✅ Valutazione completata!")