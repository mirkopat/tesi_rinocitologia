# tools/fix_filenames.py
# Corregge i file_name nelle annotazioni rimuovendo il prefisso 'images/'

import json
from pathlib import Path

def fix_filenames(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    fixed = 0
    for img in data['images']:
        old_name = img['file_name']
        # Rimuovi il prefisso 'images/' se presente
        if old_name.startswith('images/'):
            img['file_name'] = old_name[len('images/'):]
            fixed += 1
    
    with open(json_file, 'w') as f:
        json.dump(data, f)
    
    print(f"✅ Corretti {fixed} file_name in: {json_file}")
    print(f"   Esempio: {data['images'][0]['file_name']}")

if __name__ == '__main__':
    fix_filenames('dataset/annotations/instances_train.json')
    fix_filenames('dataset/annotations/instances_val.json')
    print("✅ Tutti i file_name corretti!")