# test/test_inference_debug.py

from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import torch
import json
from pathlib import Path
from PIL import Image

MODEL_PATH = "results/rtdetr/final_model"

model = RTDetrForObjectDetection.from_pretrained(MODEL_PATH)
processor = RTDetrImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

# Carica immagine
img_path = Path("data/NMCD.coco/valid/images/img_00089_jpg.rf.1b0e0a418d0e57bbfd7d1e9f7e0f5e1e.jpg")
image = Image.open(img_path).convert('RGB')

# Processa
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Ispeziona output grezzo
print("🔍 OUTPUT GREZZO:")
print(f"   logits shape: {outputs.logits.shape}")
print(f"   pred_boxes shape: {outputs.pred_boxes.shape}")
print(f"   logits values (prime 10): {outputs.logits[0, :10, :].max(dim=1).values}")

# Applica threshold manuale
logits = outputs.logits[0]  # [num_queries, num_classes]
scores = torch.sigmoid(logits)  # Probabilità
max_scores, max_labels = scores.max(dim=1)

# Trova predizioni con score > 0.01
keep = max_scores > 0.01
num_pred = keep.sum().item()
print(f"\n🔍 Predizioni con score > 0.01: {num_pred}")
if num_pred > 0:
    print(f"   Scores: {max_scores[keep]}")
    print(f"   Labels: {max_labels[keep]}")
else:
    print("   Nessuna predizione con score > 0.01")
    print(f"   Score massimo assoluto: {max_scores.max().item():.6f}")