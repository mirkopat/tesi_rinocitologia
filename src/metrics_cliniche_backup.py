# src/metrics_cliniche.py
# 
# SCOPO: Metriche cliniche per la citologia nasale.
# AUTORE: Mirko Patruno
# DATA: Settembre 2026
#
# RIFERIMENTI:
# - Macchi et al., "Standardization of Nasal Cytology: An Expert-based Delphi Consensus", 2026
# - Gelardi, "Nasal cytology in the rhinology-allergy clinic", 2025

# ============================================================================
# 1. GRIGLIA SEMI-QUANTITATIVA AICNA (Tabella 2.13 della tesi Camporeale)
# ============================================================================

AICNA_THRESHOLDS = {
    'epithelial': {'+': (1, 100), '++': (101, 200), '+++': (201, 300), '++++': (301, float('inf'))},
    'neutrophil': {'+': (1, 20), '++': (21, 40), '+++': (41, 100), '++++': (101, float('inf'))},
    'eosinophil': {'+': (1, 5), '++': (6, 10), '+++': (11, 30), '++++': (31, float('inf'))},
    'mast_cell': {'+': (1, 5), '++': (6, 10), '+++': (11, 30), '++++': (31, float('inf'))},
    'lymphocyte': {'+': (1, 5), '++': (6, 10), '+++': (11, 30), '++++': (31, float('inf'))},
    'goblet_cell': {'+': (1, 100), '++': (101, 200), '+++': (201, 300), '++++': (301, float('inf'))},
    'metaplastic': {'+': (1, 100), '++': (101, 200), '+++': (201, 300), '++++': (301, float('inf'))},
    'ciliated': {'+': (1, 100), '++': (101, 200), '+++': (201, 300), '++++': (301, float('inf'))},
    'erythrocyte': {'+': (1, 100), '++': (101, 200), '+++': (201, 300), '++++': (301, float('inf'))},
    'artifact': {'+': (1, 100), '++': (101, 200), '+++': (201, 300), '++++': (301, float('inf'))},
}

# ============================================================================
# 2. FUNZIONI PER LE METRICHE CLINICHE
# ============================================================================

def get_aicna_grade(cell_type: str, count: int) -> str:
    """
    Calcola il grado semi-quantitativo AICNA per un tipo di cellula.
    
    Args:
        cell_type (str): Tipo di cellula (es. 'eosinophil', 'neutrophil')
        count (int): Conteggio assoluto della cellula
    
    Returns:
        str: Grado AICNA ('0', '+', '++', '+++', '++++')
    """
    thresholds = AICNA_THRESHOLDS.get(cell_type)
    if thresholds is None:
        return '0'
    
    if count == 0:
        return '0'
    
    for grade, (low, high) in thresholds.items():
        if low <= count <= high:
            return grade
    return '++++'

def classify_endotype(cell_counts: dict) -> str:
    """
    Classifica l'endotipo del paziente in base ai conteggi cellulari.
    
    Args:
        cell_counts (dict): Conteggi per tipo di cellula
    
    Returns:
        str: Endotipo classificato
    """
    total = sum(cell_counts.values())
    if total == 0:
        return 'normal'
    
    eos = cell_counts.get('eosinophil', 0)
    mast = cell_counts.get('mast_cell', 0)
    neut = cell_counts.get('neutrophil', 0)
    
    eos_pct = eos / total
    mast_pct = mast / total
    neut_pct = neut / total
    
    # Soglie diagnostiche (da Gelardi, 2025)
    if eos_pct > 0.10:
        return 'allergic_rhinitis'
    elif eos_pct > 0.05 and mast_pct > 0.05:
        return 'naresma'
    elif eos_pct > 0.05:
        return 'nares'
    elif mast_pct > 0.05:
        return 'narma'
    elif neut_pct > 0.10:
        return 'narne'
    else:
        return 'normal'

def create_virtual_patient(annotations, images, cat_map, n_samples=50):
    """
    Crea un paziente virtuale aggregando N campioni microscopici.
    
    Args:
        annotations (list): Lista di annotazioni COCO
        images (list): Lista di informazioni sulle immagini
        cat_map (dict): Mappa ID categoria -> nome
        n_samples (int): Numero di campi da aggregare
    
    Returns:
        dict: Conteggi aggregati per tipo di cellula
    """
    patient_images = images[:n_samples]
    image_ids = [img['id'] for img in patient_images]
    
    counts = {}
    for ann in annotations:
        if ann['image_id'] in image_ids:
            cat_name = cat_map[ann['category_id']]
            counts[cat_name] = counts.get(cat_name, 0) + 1
    
    return counts