# src/convert_coco_to_yolo.py (VERSIONE CORRETTA)

import json
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(coco_json_path, output_dir):
    """
    Converte un file di annotazioni COCO in formato YOLO.
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # --- MAPPA FISSA DELLE CLASSI (basata sul dataset) ---
    # COCO ID → YOLO ID (escludendo 'cells')
    CLASS_MAPPING = {
        0: None,   # cells → escluso
        1: 9,      # artefatto → artifact
        2: 8,      # emazia → erythrocyte
        3: 2,      # eosinophil → eosinophil
        4: 0,      # epithelial → epithelial
        5: 7,      # epithelial ciliated → ciliated
        6: 4,      # lymphocyte → lymphocyte
        7: 3,      # mast cell → mast_cell
        8: 6,      # metaplastic → metaplastic
        9: 5,      # muciparous → goblet_cell
        10: 1,     # neutrophil → neutrophil
    }
    
    # Dizionario per le annotazioni per immagine
    img_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # Dizionario per le dimensioni delle immagini
    img_sizes = {}
    for img in coco_data['images']:
        img_sizes[img['id']] = (img['width'], img['height'])
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for img in tqdm(coco_data['images'], desc=f"Convertendo {Path(output_dir).parent.name}"):
        img_id = img['id']
        img_width, img_height = img_sizes[img_id]
        
        img_filename = Path(img['file_name']).stem
        txt_path = Path(output_dir) / f"{img_filename}.txt"
        
        annotations = img_annotations.get(img_id, [])
        
        with open(txt_path, 'w') as f:
            for ann in annotations:
                coco_id = ann['category_id']
                yolo_id = CLASS_MAPPING.get(coco_id)
                
                if yolo_id is None:  # Salta 'cells'
                    continue
                
                x, y, w, h = ann['bbox']
                
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width = w / img_width
                height = h / img_height
                
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"{yolo_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"✅ Conversione completata per {Path(coco_json_path).parent.name}")

def main():
    data_dir = Path('data/NMCD.coco')
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        coco_path = data_dir / split / '_annotations.coco.json'
        labels_dir = data_dir / split / 'labels'
        
        if coco_path.exists():
            convert_coco_to_yolo(coco_path, labels_dir)
        else:
            print(f"⚠️ File non trovato: {coco_path}")

if __name__ == '__main__':
    main()