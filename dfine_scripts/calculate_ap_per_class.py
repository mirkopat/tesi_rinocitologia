# D-FINE/calculate_ap_per_class.py
# Calcola AP per classe per D-FINE usando pycocotools

import json
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from src.core import YAMLConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- CONFIGURAZIONE ---
CONFIG_PATH = "configs/dfine/custom/dfine_hgnetv2_s_custom.yml"
CHECKPOINT_PATH = "output/dfine_hgnetv2_s_custom/best_stg1.pth"
DATA_DIR = "../data/NMCD.coco"
CONFIDENCE_THRESHOLD = 0.6

# --- CARICA MODELLO ---
cfg = YAMLConfig(CONFIG_PATH, resume=CHECKPOINT_PATH)
if "HGNetv2" in cfg.yaml_cfg:
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
cfg.model.load_state_dict(state)

class Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(cfg).to(device)
model.eval()

# --- CARICA ANNOTAZIONI ---
ann_file = Path(DATA_DIR) / "valid" / "_annotations.coco.json"
with open(ann_file, 'r') as f:
    coco_data = json.load(f)

# Salva il ground truth in un file temporaneo per COCO
with open('temp_gt.json', 'w') as f:
    json.dump(coco_data, f)

# --- GENERA PREDIZIONI IN FORMATO COCO ---
transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
predictions = []

for img in tqdm(coco_data['images'], desc="Inferenza"):
    img_path = Path(DATA_DIR) / "valid" / "images" / Path(img['file_name']).name
    if not img_path.exists():
        continue
    
    im_pil = Image.open(img_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)
    im_data = transforms(im_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)
    
    scr = scores[0]
    lab = labels[0][scr > CONFIDENCE_THRESHOLD]
    box = boxes[0][scr > CONFIDENCE_THRESHOLD]
    scrs = scr[scr > CONFIDENCE_THRESHOLD]
    
    for j, b in enumerate(box):
        x1, y1, x2, y2 = b.tolist()
        predictions.append({
            'image_id': img['id'],
            'category_id': int(lab[j].item()) + 1,  # +1 perché COCO parte da 1
            'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, w, h]
            'score': float(scrs[j].item())
        })

with open('temp_dt.json', 'w') as f:
    json.dump(predictions, f)

# --- VALUTA CON COCOEVAL ---
coco_gt = COCO('temp_gt.json')
coco_dt = coco_gt.loadRes('temp_dt.json')
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# --- ESTRAI AP PER CLASSE ---
print("\n📊 AP per classe (IoU=0.50):")
category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
for i, cat_id in enumerate(coco_gt.getCatIds()):
    ap = coco_eval.eval['precision'][:, :, i, 0, 2]  # AP@0.5
    ap_value = ap[ap > -1].mean() if len(ap[ap > -1]) > 0 else 0.0
    print(f"  {category_names[cat_id]}: {ap_value:.3f}")

# Pulizia
os.remove('temp_gt.json')
os.remove('temp_dt.json')