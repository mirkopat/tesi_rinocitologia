# tools/fix_categories.py
# Corregge i nomi delle categorie nel file JSON

import json
from pathlib import Path

def fix_categories(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Mappa delle correzioni
    corrections = {
        'epitheliall': 'epithelial',
        'muciparous': 'muciparous',
        'mast cell': 'mast_cell',
        'epithelial ciliated': 'ciliated',
    }
    
    for cat in data['categories']:
        old_name = cat['name']
        if old_name in corrections:
            cat['name'] = corrections[old_name]
            print(f"  Corretto: '{old_name}' -> '{cat['name']}'")
    
    with open(json_file, 'w') as f:
        json.dump(data, f)
    
    print(f"✅ Corretto: {json_file}")
    print(f"   Categorie: {[c['name'] for c in data['categories']]}")

if __name__ == '__main__':
    fix_categories('dataset/annotations/instances_train.json')
    fix_categories('dataset/annotations/instances_val.json')