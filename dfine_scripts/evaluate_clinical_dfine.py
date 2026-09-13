# D-FINE/evaluate_clinical_dfine.py
#
# SCOPO: Valutazione clinica di D-FINE sul dataset NMCD.
# AUTORE: Mirko Patruno
# DATA: Settembre 2026
#
# Questo script adatta la pipeline di inferenza ufficiale di D-FINE
# (tools/inference/torch_inf.py) per calcolare le metriche cito-cliniche
# sul validation set del dataset NMCD.

import json
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm import tqdm
import sys
import os

# Aggiungi la cartella principale del progetto D-FINE al path di Python
# per poter importare i moduli da 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from src.core import YAMLConfig

# ============================================================================
# 1. CONFIGURAZIONE
# ============================================================================

CONFIG_PATH = "configs/dfine/custom/dfine_hgnetv2_s_custom.yml"
CHECKPOINT_PATH = "output/dfine_hgnetv2_s_custom/best_stg1.pth"
DATA_DIR = "../data/NMCD.coco"
CONFIDENCE_THRESHOLD = 0.6 # buon bilanciamento

# Mappa dei nomi: COCO originale -> nome clinico
CLASS_NAME_MAP = {
    'epithelial': 'epithelial',
    'neutrophil': 'neutrophil',
    'eosinophil': 'eosinophil',
    'mast_cell': 'mast_cell',
    'lymphocyte': 'lymphocyte',
    'muciparous': 'goblet_cell',
    'metaplastic': 'metaplastic',
    'ciliated': 'ciliated',
    'emazia': 'erythrocyte',
    'artefatto': 'artifact'
}

CLINICAL_CLASSES = ['epithelial', 'neutrophil', 'eosinophil', 'mast_cell',
                    'lymphocyte', 'goblet_cell', 'metaplastic', 'ciliated']

# ============================================================================
# 2. CARICA MODELLO E POSTPROCESSOR (PIPELINE CORRETTA)
# ============================================================================

print(f"🔍 Caricamento D-FINE da: {CHECKPOINT_PATH}")

# Carica la configurazione
cfg = YAMLConfig(CONFIG_PATH, resume=CHECKPOINT_PATH)

# Gestione dei pesi pre-addestrati del backbone
if "HGNetv2" in cfg.yaml_cfg:
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

# Carica lo state_dict dal checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
if "ema" in checkpoint:
    state = checkpoint["ema"]["module"]
else:
    state = checkpoint["model"]

# Carica lo stato nel modello di training
cfg.model.load_state_dict(state)

# Definisci un wrapper per l'inferenza, come in torch_inf.py
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Converti modello e postprocessor in modalità 'deploy'
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(cfg).to(device)
model.eval()

print(f"   Dispositivo: {device}")

# ============================================================================
# 3. CARICA ANNOTAZIONI
# ============================================================================

ann_file = Path(DATA_DIR) / "valid" / "_annotations.coco.json"
with open(ann_file, 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']
# Mappa le categorie, escludendo 'cells'
categories = {cat['id']: cat['name'] for cat in coco_data['categories'] if cat['name'] != 'cells'}
# Crea una lista ordinata dei nomi delle classi per indicizzare i label di D-FINE
category_names_ordered = [categories[i] for i in sorted(categories.keys())]

print(f"   Immagini: {len(images)}")
print(f"   Annotazioni: {len(annotations)}")
print(f"   Classi: {len(categories)}")

anns_by_image = {}
for ann in annotations:
    img_id = ann['image_id']
    if img_id not in anns_by_image:
        anns_by_image[img_id] = []
    anns_by_image[img_id].append(ann)

# ============================================================================
# 4. VALUTAZIONE
# ============================================================================

gt_counts = Counter()
pred_counts = Counter()

print("\n📊 Valutazione su 50 immagini (paziente virtuale)...")

# Definisci le trasformazioni come in torch_inf.py
transforms = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])

for img in tqdm(images, desc="Inferenza"):
    img_path = Path(DATA_DIR) / "valid" / "images" / Path(img['file_name']).name
    
    if not img_path.exists():
        continue
    
    # Ground truth
    for ann in anns_by_image.get(img['id'], []):
        cat_id = ann['category_id']
        if cat_id in categories:
            cat_name = categories[cat_id]
            cat_name = CLASS_NAME_MAP.get(cat_name, cat_name)
            if cat_name in CLINICAL_CLASSES:
                gt_counts[cat_name] += 1
    
    # Predizione con la pipeline corretta
    im_pil = Image.open(img_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    
    # Inferenza
    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)
    
    # Estrai predizioni
    scr = scores[0]
    lab = labels[0][scr > CONFIDENCE_THRESHOLD]
    
    for label_tensor in lab:
        label_idx = label_tensor.item()
        if label_idx < len(category_names_ordered):
            cat_name_orig = category_names_ordered[label_idx]
            cat_name = CLASS_NAME_MAP.get(cat_name_orig, cat_name_orig)
            
            if cat_name in CLINICAL_CLASSES:
                pred_counts[cat_name] += 1

# ============================================================================
# 5. RISULTATI
# ============================================================================

print("\n📊 Risultati (solo classi cliniche):")
print(f"   Ground truth totali: {sum(gt_counts.values())}")
print(f"   Predizioni totali: {sum(pred_counts.values())}")

if sum(gt_counts.values()) > 0:
    print(f"   Recall approssimativo: {sum(pred_counts.values())/sum(gt_counts.values()):.2%}")

print("\n   Distribuzione per classe clinica:")
for name in CLINICAL_CLASSES:
    gt = gt_counts.get(name, 0)
    pred = pred_counts.get(name, 0)
    if gt > 0 or pred > 0:
        print(f"     {name}: pred={pred}, gt={gt}")

print("\n✅ Valutazione completata!")
