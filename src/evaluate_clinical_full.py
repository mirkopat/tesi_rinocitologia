# src/evaluate_clinical_full.py
#
# SCOPO: Valutazione clinica completa (endotipo + gradi AICNA) per YOLO e RF-DETR.
# AUTORE: Mirko Patruno
# DATA: Settembre 2026
#
# Questo script unifica la valutazione clinica per YOLO e RF-DETR.
# Per D-FINE, usare lo script dedicato in D-FINE/evaluate_clinical_dfine_full.py
#
# Calcola:
# 1. Conteggi per classe (ground truth e predetti)
# 2. Endotipo predetto (NARES, NARMA, NARNE, NARESMA)
# 3. Gradi AICNA per ogni citotipo e confronto con il ground truth

import json
import sys
import os
from pathlib import Path
from collections import Counter

# ============================================================================
# 0. GESTIONE DEI PATH
# ============================================================================

# Aggiungi la cartella principale del progetto al path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Importa le metriche cliniche
from src.metrics_cliniche import classify_endotype, get_aicna_grade

# ============================================================================
# 1. CONFIGURAZIONE
# ============================================================================

# Mappa dei nomi: COCO originale -> nome clinico
CLASS_NAME_MAP = {
    'epithelial': 'epithelial',
    'neutrophil': 'neutrophil',
    'eosinophil': 'eosinophil',
    'mast cell': 'mast_cell',
    'mast_cell': 'mast_cell',
    'lymphocyte': 'lymphocyte',
    'muciparous': 'goblet_cell',
    'metaplastic': 'metaplastic',
    'epithelial ciliated': 'ciliated',
    'ciliated': 'ciliated',
    'emazia': 'erythrocyte',
    'artefatto': 'artifact'
}

# Classi cliniche (escluse emazia e artefatto)
CLINICAL_CLASSES = ['epithelial', 'neutrophil', 'eosinophil', 'mast_cell',
                    'lymphocyte', 'goblet_cell', 'metaplastic', 'ciliated']


# ============================================================================
# 2. FUNZIONI DI STAMPA
# ============================================================================

def print_counts_comparison(gt_counts, pred_counts):
    """Stampa il confronto dei conteggi."""
    print("\n" + "="*70)
    print("CONFRONTO CONTEGGI")
    print("="*70)
    print(f"{'Citotipo':<20} {'GT':<10} {'Pred':<10} {'Differenza':<15}")
    print("-"*70)
    
    for cell_type in CLINICAL_CLASSES:
        gt = gt_counts.get(cell_type, 0)
        pred = pred_counts.get(cell_type, 0)
        diff = pred - gt
        sign = "+" if diff > 0 else ""
        print(f"{cell_type:<20} {gt:<10} {pred:<10} {sign}{diff:<15}")


def print_grades_comparison(gt_counts, pred_counts):
    """Stampa il confronto dei gradi AICNA tra GT e predizioni."""
    print("\n" + "="*70)
    print("CONFRONTO GRADI AICNA")
    print("="*70)
    print(f"{'Citotipo':<20} {'GT':<8} {'Pred':<8} {'Match':<8}")
    print("-"*70)
    
    correct_grades = 0
    total_grades = 0
    
    for cell_type in CLINICAL_CLASSES:
        gt_count = gt_counts.get(cell_type, 0)
        pred_count = pred_counts.get(cell_type, 0)
        
        gt_grade = get_aicna_grade(cell_type, gt_count)
        pred_grade = get_aicna_grade(cell_type, pred_count)
        
        match = "OK" if gt_grade == pred_grade else "NO"
        if gt_grade == pred_grade:
            correct_grades += 1
        total_grades += 1
        
        print(f"{cell_type:<20} {gt_grade:<8} {pred_grade:<8} {match:<8}")
    
    accuracy = correct_grades / total_grades if total_grades > 0 else 0.0
    print("-"*70)
    print(f"Accuratezza gradi AICNA: {correct_grades}/{total_grades} = {accuracy:.2%}")
    
    return accuracy


def print_endotype_comparison(gt_counts, pred_counts):
    """Stampa il confronto degli endotipi."""
    gt_endotype = classify_endotype(gt_counts)
    pred_endotype = classify_endotype(pred_counts)
    match = "OK" if gt_endotype == pred_endotype else "NO"
    
    print("\n" + "="*70)
    print("CONFRONTO ENDOTIPI")
    print("="*70)
    print(f"Ground Truth: {gt_endotype}")
    print(f"Predetto:     {pred_endotype}  [{match}]")
    print("="*70)
    
    return gt_endotype == pred_endotype


def print_summary(gt_counts, pred_counts):
    """Stampa il riepilogo completo."""
    print("\n" + "="*70)
    print("RIEPILOGO VALUTAZIONE CLINICA")
    print("="*70)
    
    gt_total = sum(gt_counts.values())
    pred_total = sum(pred_counts.values())
    
    print(f"Ground truth totali: {gt_total}")
    print(f"Predizioni totali:   {pred_total}")
    if gt_total > 0:
        print(f"Recall approssimativo: {pred_total/gt_total:.2%}")
    
    # Confronto conteggi
    print_counts_comparison(gt_counts, pred_counts)
    
    # Confronto gradi AICNA
    grade_accuracy = print_grades_comparison(gt_counts, pred_counts)
    
    # Confronto endotipi
    endotype_correct = print_endotype_comparison(gt_counts, pred_counts)
    
    # Riepilogo finale
    print("\n" + "="*70)
    print("CONCLUSIONI")
    print("="*70)
    print(f"Endotipo corretto:    {'SI' if endotype_correct else 'NO'}")
    print(f"Accuratezza gradi:    {grade_accuracy:.2%}")
    print("="*70)
    
    return {
        'endotype_correct': endotype_correct,
        'grade_accuracy': grade_accuracy,
        'gt_total': gt_total,
        'pred_total': pred_total
    }


# ============================================================================
# 3. FUNZIONI PER YOLO E RF-DETR
# ============================================================================

def evaluate_yolo(model_path, data_dir, split='valid', conf=0.25):
    """Valuta un modello YOLO."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    ann_file = Path(data_dir) / split / '_annotations.coco.json'
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    cat_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Ground truth
    gt_counts = Counter()
    for ann in coco_data['annotations']:
        orig_name = cat_map[ann['category_id']]
        yolo_name = CLASS_NAME_MAP.get(orig_name)
        if yolo_name and yolo_name in CLINICAL_CLASSES:
            gt_counts[yolo_name] += 1
    
    # Predizioni
    pred_counts = Counter()
    for img in coco_data['images']:
        img_name = Path(img['file_name']).stem + '.jpg'
        if '.rf.' in img_name:
            img_name = img_name.split('.rf.')[0] + '.jpg'
        
        img_path = Path(data_dir) / split / 'images' / img_name
        if not img_path.exists():
            img_path = Path(data_dir) / split / 'images' / Path(img['file_name']).name
        
        if img_path.exists():
            results = model(img_path, conf=conf, verbose=False)
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    yolo_name = results[0].names[class_id]
                    if yolo_name in CLINICAL_CLASSES:
                        pred_counts[yolo_name] += 1
    
    return gt_counts, pred_counts


def evaluate_rfdetr(model_path, data_dir, split='valid', conf=0.6):
    """Valuta un modello RF-DETR."""
    from rfdetr import RFDETRBase
    from PIL import Image
    from tqdm import tqdm
    
    model = RFDETRBase(pretrain_weights=f"{model_path}/checkpoint_best_ema.pth")
    
    ann_file = Path(data_dir) / split / '_annotations.coco.json'
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories'] if cat['id'] != 0}
    
    anns_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)
    
    gt_counts = Counter()
    pred_counts = Counter()
    
    for img in tqdm(coco_data['images'], desc="Inferenza RF-DETR"):
        img_path = Path(data_dir) / split / 'images' / Path(img['file_name']).name
        if not img_path.exists():
            continue
        
        # Ground truth
        for ann in anns_by_image.get(img['id'], []):
            cat_id = ann['category_id']
            if cat_id != 0 and cat_id in categories:
                cat_name_orig = categories[cat_id]
                cat_name = CLASS_NAME_MAP.get(cat_name_orig, cat_name_orig)
                if cat_name in CLINICAL_CLASSES:
                    gt_counts[cat_name] += 1
        
        # Predizione
        image = Image.open(img_path).convert('RGB')
        detections = model.predict(image, threshold=conf)
        
        if hasattr(detections, 'class_id') and hasattr(detections, 'confidence'):
            if hasattr(detections, 'data') and 'class_name' in detections.data:
                class_names = detections.data['class_name']
                for idx, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
                    if confidence > conf and idx < len(class_names):
                        cat_name_orig = class_names[idx]
                        cat_name = CLASS_NAME_MAP.get(cat_name_orig, cat_name_orig)
                        if cat_name in CLINICAL_CLASSES:
                            pred_counts[cat_name] += 1
    
    return gt_counts, pred_counts


# ============================================================================
# 4. MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Valutazione clinica completa (YOLO e RF-DETR)")
    parser.add_argument('--model', type=str, required=True, choices=['yolo', 'rfdetr'])
    parser.add_argument('--path', type=str, required=True, help="Percorso del modello")
    parser.add_argument('--data-dir', type=str, default='data/NMCD.coco')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--conf', type=float, default=None)
    
    args = parser.parse_args()
    
    print(f"Valutazione {args.model.upper()}...")
    print(f"   Modello: {args.path}")
    
    if args.model == 'yolo':
        conf = args.conf if args.conf else 0.25
        gt_counts, pred_counts = evaluate_yolo(args.path, args.data_dir, args.split, conf)
    elif args.model == 'rfdetr':
        conf = args.conf if args.conf else 0.6
        gt_counts, pred_counts = evaluate_rfdetr(args.path, args.data_dir, args.split, conf)
    
    # Stampa riepilogo
    results = print_summary(gt_counts, pred_counts)