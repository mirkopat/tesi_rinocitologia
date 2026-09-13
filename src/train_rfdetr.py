# src/train_rfdetr.py
#
# SCOPO: Addestramento di RF-DETR sul dataset NMCD.
# AUTORE: Mirko Patruno
# DATA: Settembre 2026
#
# RF-DETR è un modello Transformer-based per object detection.
# Supporta nativamente il formato COCO e YOLO, con rilevamento automatico.

from rfdetr import RFDETRBase

if __name__ == '__main__':
    # Il modello rileva automaticamente il formato COCO
    model = RFDETRBase()

    # Avvia l'addestramento
    model.train(
    dataset_dir="data/NMCD.coco",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    num_workers=0,
    output_dir="results/rfdetr_100ep",
    resolution=336,
)