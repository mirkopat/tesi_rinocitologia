# src/fix_coco_paths.py
# Corregge i percorsi delle immagini nei file COCO per RF-DETR

import json
from pathlib import Path

def fix_paths(coco_file):
    with open(coco_file, 'r') as f:
        coco = json.load(f)
    
    # Aggiungi "images/" al prefisso di ogni file_name
    for img in coco['images']:
        if not img['file_name'].startswith('images/'):
            img['file_name'] = 'images/' + img['file_name']
    
    with open(coco_file, 'w') as f:
        json.dump(coco, f)
    
    print(f"✅ Corretto: {coco_file}")

# Correggi tutti gli split
fix_paths('data/NMCD.coco/train/_annotations.coco.json')
fix_paths('data/NMCD.coco/valid/_annotations.coco.json')
fix_paths('data/NMCD.coco/test/_annotations.coco.json')

print("✅ Tutti i percorsi corretti!")