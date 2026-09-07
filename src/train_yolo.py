# src/train_yolo.py
# 
# SCOPO: Addestramento del modello YOLOv10 sul dataset NMCD.
# AUTORE: Mirko Patruno
# DATA: Settembre 2026

from ultralytics import YOLO
import torch

def main():
    """Funzione principale per l'addestramento"""
    
    # Verifica se la GPU è disponibile
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔍 Dispositivo utilizzato: {device}")
    
    if device == 'cpu':
        print("⚠️  ATTENZIONE: Stai usando la CPU. L'addestramento sarà molto lento.")
    
    # --- CONFIGURAZIONE ---
    DATA_YAML_PATH = 'data/NMCD.coco/data.yaml'
    MODEL_SIZE = 'n'      # 'n' (nano) - più veloce
    EPOCHS = 50
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    OUTPUT_DIR = 'results/yolo10'
    
    print(f"🚀 Avvio addestramento YOLOv10{MODEL_SIZE}...")
    print(f"   Epoche: {EPOCHS}, Batch: {BATCH_SIZE}, Img Size: {IMAGE_SIZE}")
    
    # --- CARICAMENTO MODELLO ---
    model = YOLO(f'yolov10{MODEL_SIZE}.pt') 
    
    # --- ADDESTRAMENTO ---
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=device,
        workers=0,                # ← IMPORTANTE: 0 invece di 4 su Windows!
        patience=10,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        project=OUTPUT_DIR,
        name='exp',
        exist_ok=True,
        save=True,
        save_period=10,
        plots=True,
    )
    
    print("✅ Addestramento completato!")
    print(f"📁 Modello salvato in: {OUTPUT_DIR}")

# Punto di ingresso principale (necessario su Windows!)
if __name__ == '__main__':
    main()