# tools/convert_annotations.py
# Converte i category_id delle annotazioni COCO per D-FINE
# Da: 1,2,3,...,10 -> A: 0,1,2,...,9

import json
import sys
from pathlib import Path

def convert_annotations(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Crea una mappa: vecchio category_id -> nuovo category_id (da 0)
    # Escludi la categoria 'cells' (id 0) se presente
    old_categories = [cat for cat in data['categories'] if cat['name'] != 'cells']
    old_ids = sorted([cat['id'] for cat in old_categories])
    
    id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    
    print(f"  Mappatura: {id_map}")
    
    # Aggiorna le annotazioni
    new_annotations = []
    for ann in data['annotations']:
        old_id = ann['category_id']
        if old_id in id_map:
            ann['category_id'] = id_map[old_id]
            new_annotations.append(ann)
    
    # Aggiorna le categorie
    new_categories = []
    for new_id, old_id in enumerate(old_ids):
        for cat in data['categories']:
            if cat['id'] == old_id:
                new_cat = cat.copy()
                new_cat['id'] = new_id
                new_categories.append(new_cat)
                break
    
    data['annotations'] = new_annotations
    data['categories'] = new_categories
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"  ✅ Convertito: {output_file}")
    print(f"     - Immagini: {len(data['images'])}")
    print(f"     - Annotazioni: {len(data['annotations'])}")
    print(f"     - Categorie: {[c['name'] for c in data['categories']]}")

if __name__ == '__main__':
    print("🔧 Conversione annotazioni per D-FINE")
    convert_annotations('dataset/annotations/instances_train.json', 
                        'dataset/annotations/instances_train_converted.json')
    convert_annotations('dataset/annotations/instances_val.json', 
                        'dataset/annotations/instances_val_converted.json')
    print("✅ Tutte le annotazioni convertite!")