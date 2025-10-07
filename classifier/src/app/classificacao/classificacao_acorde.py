# src/app/classificacao/classificacao_acorde.py
from typing import List
from .acorde_resultado import AcordeResultado
import numpy as np

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Templates defined in pitch-class indices (0=C .. 11=B)
TEMPLATES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "5":    [0, 7],
    "7":    [0, 4, 7, 10],   # dominant 7
    "maj7": [0, 4, 7, 11],   # major 7
    "m7":   [0, 3, 7, 10],   # minor 7
}

def pc_vector_from_offsets(offsets):
    v = np.zeros(12, dtype=float)
    for o in offsets:
        v[o % 12] = 1.0
    return v

TEMPLATE_VECS = {k: pc_vector_from_offsets(v) for k, v in TEMPLATES.items()}

def best_template_match(pitch_class_vector: np.ndarray):
    """
    pitch_class_vector: length-12 non-negative (can be normalized)
    Returns: (best_name, best_root_index, best_score)
    """
    if np.linalg.norm(pitch_class_vector) == 0:
        return ("Nenhum", None, 0.0)

    vnorm = pitch_class_vector / (np.linalg.norm(pitch_class_vector) + 1e-9)
    best = (None, None, -1.0)
    for root in range(12):
        for tname, tvec in TEMPLATE_VECS.items():
            # shift template so that template root aligns with 'root'
            templ = np.roll(tvec, root)
            tnorm = templ / (np.linalg.norm(templ) + 1e-9)
            score = float(np.dot(vnorm, tnorm))  # cosine because normalized
            if score > best[2]:
                best = (tname, root, score)
    # name like Amaj7 etc.
    tname, root, score = best
    if root is None:
        return ("Indefinido", None, 0.0)
    full_name = NOTE_NAMES[root] + (tname if tname != "5" else "5")
    return (full_name, root, float(score))

def classify_chord_from_pitchclass_vector(pc_vector: np.ndarray):
    from .acorde_resultado import AcordeResultado
    """
    pc_vector: length-12 numeric vector (not necessarily normalized). 
    Returns AcordeResultado-like dict.
    """
    thr = 0.08
    pc_vector = np.where(pc_vector >= thr, pc_vector, 0.0)
    name, root, score = best_template_match(pc_vector)
    # For compatibility with AcordeResultado dataclass in your project:
    pcs_present = [NOTE_NAMES[i] for i in range(12) if pc_vector[i] > 0]
    return AcordeResultado(nome=name, confianca=score, notas=pcs_present)
