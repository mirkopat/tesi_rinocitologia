# test/test_inference.py (modificato)

import sys
import json
from pathlib import Path
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
from PIL import Image
import torch

# Argomento: checkpoint opzionale
checkpoint = sys.argv[1] if len(sys.argv) > 1 else "results/rtdetr/final_model"

DATA_DIR = "data/NMCD.coco"
SPLIT = "valid"

# Trova la prima immagine
ann_file = Path(DATA_DIR) / SPLIT / '_annotations.coco.json'
with open(ann_file, 'r') as f:
    coco_data = json.load(f)

first_image = coco_data['images'][0]
img_path = Path(DATA_DIR) / SPLIT / 'images' / first_image['file_name']

print(f"🔍 Percorso immagine: {img_path}")
print(f"   Esiste? {img_path.exists()}")
print(f"🔍 Caricamento modello da: {checkpoint}")

try:
    model = RTDetrForObjectDetection.from_pretrained(checkpoint)
    processor = RTDetrImageProcessor.from_pretrained(checkpoint)
except Exception as e:
    print(f"❌ Errore nel caricamento: {e}")
    exit()

model.eval()

image = Image.open(img_path).convert('RGB')
print(f"   Immagine caricata: {image.size}")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

print(f"\n🔍 Box rilevati: {len(results['boxes'])}")
if len(results['boxes']) > 0:
    print(f"   Scores: {results['scores'][:5]}")
    print(f"   Labels: {results['labels'][:5]}")
    print(f"\n✅ Il modello funziona!")
else:
    print(f"⚠️ Nessuna rilevazione!")
    
    # Debug: stampa gli output grezzi del modello
    print("\n🔧 DEBUG - Output grezzi:")
    print(f"   logits shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")
    print(f"   pred_boxes shape: {outputs.pred_boxes.shape if hasattr(outputs, 'pred_boxes') else 'N/A'}")
    if hasattr(outputs, 'logits'):
        print(f"   logits max: {outputs.logits.max().item():.3f}")
        print(f"   logits min: {outputs.logits.min().item():.3f}")