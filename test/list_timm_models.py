# src/list_timm_models.py
import timm

print("🔍 Modelli disponibili in timm:")
print("=" * 50)

# Cerca tutti i modelli che contengono "det"
det_models = [m for m in timm.list_models() if 'det' in m.lower()]
print(f"\n📦 Modelli di object detection trovati: {len(det_models)}")
for m in sorted(det_models)[:20]:
    print(f"  - {m}")

# Cerca specificamente "efficientdet"
effdet_models = [m for m in timm.list_models() if 'efficientdet' in m.lower()]
print(f"\n📦 Modelli EfficientDet trovati: {len(effdet_models)}")
for m in sorted(effdet_models):
    print(f"  - {m}")

if len(effdet_models) == 0:
    print("\n⚠️ Nessun modello EfficientDet trovato in timm.")
    print("   Proviamo con il modulo effdet direttamente.")