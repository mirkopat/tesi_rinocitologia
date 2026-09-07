# src/train_yolo_nano_v2.py
# 
# SCOPO: Addestramento YOLOv10n con 100 epoche per migliorare le performance
# AUTORE: Mirko Patruno
# DATA: Settembre 2026

from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔍 Dispositivo utilizzato: {device}")
    
    # --- CONFIGURAZIONE ---
    DATA_YAML_PATH = 'data/NMCD.coco/data.yaml'
    MODEL_SIZE = 'n'           # Nano (veloce!)
    EPOCHS = 100               # 100 epoche
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    OUTPUT_DIR = 'results/yolo10_nano_v2'
    
    print(f"🚀 Avvio addestramento YOLOv10{MODEL_SIZE} (100 epoche)...")
    print(f"   Epoche: {EPOCHS}, Batch: {BATCH_SIZE}, Img Size: {IMAGE_SIZE}")
    print(f"   Tempo stimato: ~1.5-2 ore")
    
    model = YOLO(f'yolov10{MODEL_SIZE}.pt')
    
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=device,
        workers=0,              # Necessario su Windows
        patience=20,            # Early stopping dopo 20 epoche senza miglioramenti
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

if __name__ == '__main__':
    main()