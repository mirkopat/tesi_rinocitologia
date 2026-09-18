# src/evaluate_clinical_from_predictions.py
import argparse
import json
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics_cliniche import classify_endotype, get_aicna_grade

# ============================================================================
# 1. CONFIGURAZIONE
# ============================================================================

TESISTA_ROOT = Path("lavoro_tesista")

MODEL_FILES = {
    "transfer_learning": {
        "valid": ("transfer_valid_predictions.csv", "valid"),
        "test":  ("transfer_test_predictions.csv",  "test"),
    },
    "transfer_resnet18": {
        "valid": ("transfer_valid_predictions.csv", "valid"),
        "test":  ("transfer_test_predictions.csv",  "test"),
    },
    "cnn_residual": {
        "valid": ("cnn_valid_predictions.csv", "valid"),
        "test":  ("cnn_test_predictions.csv",  "test"),
    },
}

METADATA_FILE = TESISTA_ROOT / "outputs" / "metadata" / "cell_crops.csv"

COCO_FILES = {
    "valid": TESISTA_ROOT / "Dataset Rinocitologia" / "NMCD.coco" / "valid" / "_annotations.coco.json",
    "test":  TESISTA_ROOT / "Dataset Rinocitologia" / "NMCD.coco" / "test"  / "_annotations.coco.json",
}

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

CLINICAL_CLASSES = ['epithelial', 'neutrophil', 'eosinophil', 'mast_cell',
                    'lymphocyte', 'goblet_cell', 'metaplastic', 'ciliated']

# ============================================================================
# 2. ARGOMENTI
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODEL_FILES.keys()))
parser.add_argument("--split", required=True, choices=["valid", "test"])
parser.add_argument("--images-per-patient", type=int, default=50)
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

pred_filename, coco_split = MODEL_FILES[args.model][args.split]
PREDICTIONS_FILE = TESISTA_ROOT / "outputs" / args.model / pred_filename
COCO_FILE = COCO_FILES[coco_split]

# ============================================================================
# 3. CARICA DATI
# ============================================================================

if not args.quiet:
    print(f"\n=== {args.model} / {args.split} ===")
    print(f"Predizioni: {PREDICTIONS_FILE}")

pred_df = pd.read_csv(PREDICTIONS_FILE)
meta_df = pd.read_csv(METADATA_FILE)

with open(COCO_FILE, 'r') as f:
    coco_data = json.load(f)

if not args.quiet:
    print(f"Predizioni: {len(pred_df)} righe")
    print(f"Metadati:   {len(meta_df)} righe")

# ============================================================================
# 4. ALLINEA PREDIZIONI E METADATI (per split)
# ============================================================================

meta_split = meta_df[meta_df['split'] == coco_split].reset_index(drop=True)
if not args.quiet:
    print(f"Metadati split='{coco_split}': {len(meta_split)} righe")

if len(meta_split) != len(pred_df):
    print(f"⚠️  Righe diverse: meta={len(meta_split)}, pred={len(pred_df)}")
    n = min(len(meta_split), len(pred_df))
    pred_df = pred_df.iloc[:n].reset_index(drop=True)
    meta_split = meta_split.iloc[:n].reset_index(drop=True)

pred_df['annotation_id'] = meta_split['annotation_id'].values
pred_df['image_id']      = meta_split['image_id'].values
pred_df['true_label_meta'] = meta_split['label'].values

tl_pred = pred_df['true_label'].astype(str).str.strip()
tl_meta = pred_df['true_label_meta'].astype(str).str.strip()
mismatch = (tl_pred != tl_meta).sum()

if not args.quiet:
    print(f"Disallineamenti true_label: {mismatch}/{len(pred_df)}")

# ============================================================================
# 5. AGGREGA PER IMMAGINE
# ============================================================================

gt_by_image = defaultdict(Counter)
pred_by_image = defaultdict(Counter)

for _, row in pred_df.iterrows():
    img_id = row['image_id']
    if pd.isna(img_id):
        continue
    true_name = CLASS_NAME_MAP.get(row['true_label'], row['true_label'])
    pred_name = CLASS_NAME_MAP.get(row['predicted_label'], row['predicted_label'])
    if true_name in CLINICAL_CLASSES:
        gt_by_image[img_id][true_name] += 1
    if pred_name in CLINICAL_CLASSES:
        pred_by_image[img_id][pred_name] += 1

# ============================================================================
# 6. PAZIENTE VIRTUALE
# ============================================================================

image_ids = [img['id'] for img in coco_data['images']]
images_per_patient = args.images_per_patient
total_patients = max(1, len(image_ids) // images_per_patient)

endotype_correct = 0
grade_accuracies = []

for pidx in range(total_patients):
    s = pidx * images_per_patient
    e = s + images_per_patient
    pids = image_ids[s:e]

    gt_counts = Counter()
    pred_counts = Counter()
    for img_id in pids:
        gt_counts.update(gt_by_image.get(img_id, Counter()))
        pred_counts.update(pred_by_image.get(img_id, Counter()))

    gt_endotype = classify_endotype(dict(gt_counts))
    pred_endotype = classify_endotype(dict(pred_counts))

    if gt_endotype == pred_endotype:
        endotype_correct += 1

    matches = 0
    total = 0
    for ct in CLINICAL_CLASSES:
        g = get_aicna_grade(ct, gt_counts.get(ct, 0))
        p = get_aicna_grade(ct, pred_counts.get(ct, 0))
        if g == p:
            matches += 1
        total += 1
    grade_acc = matches / total if total > 0 else 0.0
    grade_accuracies.append(grade_acc)

    if not args.quiet:
        marker = 'OK' if gt_endotype == pred_endotype else 'NO'
        print(f"\nPaziente {pidx + 1} (immagini {s}–{e - 1}):")
        print(f"  GT:   {gt_endotype:10s}  | {dict(gt_counts)}")
        print(f"  Pred: {pred_endotype:10s}  | {dict(pred_counts)}")
        print(f"  Match endotipo: {marker}  (GT={gt_endotype}, Pred={pred_endotype})")
        print(f"  Accuratezza gradi: {grade_acc:.2%}")

        # ---- NUOVE RIGHE ----
        gt_leuk = sum(gt_counts.get(c, 0) for c in ['neutrophil','eosinophil','lymphocyte','mast_cell'])
        pred_leuk = sum(pred_counts.get(c, 0) for c in ['neutrophil','eosinophil','lymphocyte','mast_cell'])
        print(f"  Leucociti GT/Pred: {gt_leuk}/{pred_leuk}")
        if gt_leuk:
            gt_eos_pct  = gt_counts.get('eosinophil', 0) / gt_leuk
            gt_mast_pct = gt_counts.get('mast_cell', 0) / gt_leuk
            gt_neut_pct = gt_counts.get('neutrophil', 0) / gt_leuk
            print(f"  GT %:   eos={gt_eos_pct:.1%}  mast={gt_mast_pct:.1%}  neut={gt_neut_pct:.1%}")
        if pred_leuk:
            pr_eos_pct  = pred_counts.get('eosinophil', 0) / pred_leuk
            pr_mast_pct = pred_counts.get('mast_cell', 0) / pred_leuk
            pr_neut_pct = pred_counts.get('neutrophil', 0) / pred_leuk
            print(f"  Pred %: eos={pr_eos_pct:.1%}  mast={pr_mast_pct:.1%}  neut={pr_neut_pct:.1%}")

# ============================================================================
# 7. RIEPILOGO
# ============================================================================

endotype_acc = endotype_correct / total_patients if total_patients else 0.0
grade_acc_avg = sum(grade_accuracies) / len(grade_accuracies) if grade_accuracies else 0.0

print(f"\n>>> {args.model} | {args.split} | pazienti={total_patients} | "
      f"endotipo_acc={endotype_acc:.2%} | gradi_acc={grade_acc_avg:.2%}")