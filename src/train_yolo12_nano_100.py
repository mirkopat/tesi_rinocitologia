# src/train_yolo12_nano.py
#
# SCOPO: Addestramento di YOLO12 NANO sul dataset NMCD.
# AUTORE: Mirko Patruno
# DATA: Settembre 2026

from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo utilizzato: {device}")
    
    # Configurazione
    DATA_YAML_PATH = 'data/NMCD.coco/data.yaml'
    MODEL_SIZE = 'n'  # 'n', 's', 'm', 'l', 'x'
    EPOCHS = 100
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    OUTPUT_DIR = 'results/yolo12_nano_100'  # Cartella separata!
    
    print(f"Avvio addestramento YOLO12{MODEL_SIZE}...")
    print(f"   Epoche: {EPOCHS}, Batch: {BATCH_SIZE}, Img Size: {IMAGE_SIZE}")
    
    # Carica modello pre-addestrato YOLO12
    model = YOLO(f'yolo12{MODEL_SIZE}.pt')
    
    # Addestra
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=device,
        workers=0,
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
    
    print("Addestramento completato!")
    print(f"Modello salvato in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()