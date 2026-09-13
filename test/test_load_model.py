# src/test_load_model.py

from transformers import RTDetrForObjectDetection
import torch

# Prova diversi checkpoint
checkpoints = [
    "results/rtdetr/final_model",
    "results/rtdetr/checkpoint-1800",
    "results/rtdetr/checkpoint-1000",
    "results/rtdetr/checkpoint-200",
]

for ckpt in checkpoints:
    try:
        print(f"🔍 Caricamento: {ckpt}")
        model = RTDetrForObjectDetection.from_pretrained(ckpt)
        
        # Verifica che il modello abbia pesi non casuali
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Parametri totali: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        
        # Controlla se i pesi sono normali (non casuali)
        first_weight = next(model.parameters()).flatten()
        print(f"   Primo peso: {first_weight[0].item():.6f}")
        print(f"   Media pesi: {first_weight.mean().item():.6f}")
        print(f"   Deviazione std: {first_weight.std().item():.6f}")
        print(f"   ✅ OK")
        print()
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        print()