# src/check_labels.py
import os
from pathlib import Path
from collections import Counter

labels_dir = Path('data/NMCD.coco/valid/labels')

class_counts = Counter()
total_files = 0

for label_file in labels_dir.glob('*.txt'):
    total_files += 1
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                class_counts[class_id] += 1

print(f"File di labels analizzati: {total_files}")
print("\nDistribuzione delle classi (YOLO ID):")
for class_id, count in sorted(class_counts.items()):
    print(f"  Classe {class_id}: {count} istanze")