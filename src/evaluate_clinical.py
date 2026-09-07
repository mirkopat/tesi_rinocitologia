# src/evaluate_clinical.py (versione corretta)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import json
from pathlib import Path
from collections import Counter
from src.metrics_cliniche import classify_endotype, get_aicna_grade

# Mappa per convertire i nomi delle classi
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
    'artefatto': 'artifact',
    'cells': None  # Escludi 'cells' completamente
}

def evaluate_model(model_path, data_dir, split='valid'):
    """Valuta un modello e calcola le metriche cliniche."""
    model = YOLO(model_path)
    
    ann_file = Path(data_dir) / split / '_annotations.coco.json'
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    cat_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print(f"📊 Valutazione su {split} set...")
    print(f"   Immagini totali: {len(coco_data['images'])}")
    
    images_per_patient = 50
    total_patients = len(coco_data['images']) // images_per_patient
    print(f"   Pazienti virtuali: {total_patients}")
    
    all_ground_truths = []
    all_predictions = []
    
    for patient_id in range(total_patients):
        start_idx = patient_id * images_per_patient
        end_idx = start_idx + images_per_patient
        patient_images = coco_data['images'][start_idx:end_idx]
        
        print(f"\n🔬 Paziente {patient_id+1}: {len(patient_images)} immagini")
        
        # Ground Truth (converte i nomi)
        gt_counts = Counter()
        for ann in coco_data['annotations']:
            if any(img['id'] == ann['image_id'] for img in patient_images):
                orig_name = cat_map[ann['category_id']]
                yolo_name = CLASS_NAME_MAP.get(orig_name)
                if yolo_name and yolo_name not in ['erythrocyte', 'artifact']:
                    gt_counts[yolo_name] += 1
        
        # Predizioni
        pred_counts = Counter()
        for img in patient_images:
            img_name = Path(img['file_name']).stem + '.jpg'
            if '.rf.' in img_name:
                img_name = img_name.split('.rf.')[0] + '.jpg'
            
            img_path = Path(data_dir) / split / 'images' / img_name
            if not img_path.exists():
                img_path = Path(data_dir) / split / 'images' / Path(img['file_name']).name
            
            if img_path.exists():
                results = model(img_path, conf=0.25, verbose=False)
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        class_id = int(box.cls[0])
                        # Ottieni il nome della classe dal modello (YOLO)
                        yolo_name = results[0].names[class_id]
                        if yolo_name not in ['erythrocyte', 'artifact']:
                            pred_counts[yolo_name] += 1
        
        print(f"   GT: {dict(gt_counts)}")
        print(f"   Pred: {dict(pred_counts)}")
        
        all_ground_truths.append(dict(gt_counts))
        all_predictions.append(dict(pred_counts))
    
    # Calcola metriche cliniche
    endotype_acc = calculate_endotype_accuracy(all_ground_truths, all_predictions)
    
    return {
        'endotype_accuracy': endotype_acc,
        'patients': len(all_predictions)
    }

def calculate_endotype_accuracy(gt_patients, pred_patients):
    """Calcola l'accuratezza degli endotipi predetti."""
    correct = 0
    total = len(gt_patients)
    
    print("\n" + "="*50)
    print("CONFRONTO ENDOTIPI")
    print("="*50)
    
    for i, (gt, pred) in enumerate(zip(gt_patients, pred_patients)):
        gt_endotype = classify_endotype(gt)
        pred_endotype = classify_endotype(pred)
        is_correct = gt_endotype == pred_endotype
        
        print(f"Paziente {i+1}: GT={gt_endotype:15s} | Pred={pred_endotype:15s} | {'✅' if is_correct else '❌'}")
        
        if is_correct:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n📊 Accuratezza endotipi: {correct}/{total} = {accuracy:.2%}")
    
    return accuracy

if __name__ == '__main__':
    print("🔍 Valutazione YOLOv10m...")
    
    results = evaluate_model(
        model_path='runs/detect/results/yolo10_nano_v2/exp/weights/best.pt',
        data_dir='data/NMCD.coco',
        split='valid'
    )
    
    print(f"\n✅ Endotype Accuracy: {results['endotype_accuracy']:.2%}")
    print(f"   Pazienti valutati: {results['patients']}")