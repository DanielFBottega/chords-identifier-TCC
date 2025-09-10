import numpy as np
from typing import Dict, Tuple, List
from .note_utils import pc_to_name, name_pt

# Templates de qualidades (intervalos a partir da tônica)
# Representamos como sets de classes de intervalo
QUALITY_INTERVALS: Dict[str, set] = {
    "M":   {0,4,7},            # maior
    "m":   {0,3,7},            # menor
    "7":   {0,4,7,10},         # dominante
    "maj7":{0,4,7,11},         # maior com 7M
    "m7":  {0,3,7,10},         # menor com 7m
    "dim": {0,3,6},            # diminuto
    "dim7":{0,3,6,9},          # diminuto com 7º diminuta
    "m7b5":{0,3,6,10},         # meio-diminuto
    "aug": {0,4,8},            # aumentado
    "sus4":{0,5,7},            # suspenso 4
    "sus2":{0,2,7},            # suspenso 2
    "6":   {0,4,7,9},          # sexta
    "m6":  {0,3,7,9},          # menor com sexta
    "5":   {0,7},              # power chord (não é tríade, mas útil)
}

# Ordem de preferência para desempate
QUALITY_PRIORITY = ["maj7","m7","7","6","m6","M","m","sus4","sus2","aug","m7b5","dim7","dim","5"]

def rotate_set(intervals: set, shift: int) -> set:
    return { (i + shift) % 12 for i in intervals }

def chord_fit_score(pcs: set, root: int, quality: str) -> float:
    # Score = proporção de cobertura dos intervalos esperados + pequena penalização para extra notes
    expected = rotate_set(QUALITY_INTERVALS[quality], root)
    covered = len(pcs & expected) / max(1, len(expected))
    extras = len(pcs - expected)
    penalty = 0.05 * extras
    return max(0.0, covered - penalty)

def best_chord_from_pcs(pcs: List[int]) -> Tuple[int, str, float]:
    if not pcs:
        return 0, "M", 0.0
    pcs_set = set(int(p)%12 for p in pcs)
    best = (-1.0, 0, "M")
    for r in range(12):
        for q in QUALITY_PRIORITY:
            s = chord_fit_score(pcs_set, r, q)
            if s > best[0] + 1e-6:
                best = (s, r, q)
    score, root, qual = best
    return root, qual, float(score)

def chord_symbol(root_pc: int, quality: str) -> str:
    root_name = pc_to_name(root_pc)
    # Símbolos concisos
    if quality == "M": return f"{root_name}"
    if quality == "m": return f"{root_name}m"
    if quality == "7": return f"{root_name}7"
    if quality == "maj7": return f"{root_name}maj7"
    if quality == "m7": return f"{root_name}m7"
    if quality == "dim": return f"{root_name}dim"
    if quality == "dim7": return f"{root_name}dim7"
    if quality == "m7b5": return f"{root_name}m7b5"
    if quality == "aug": return f"{root_name}aug"
    if quality == "sus4": return f"{root_name}sus4"
    if quality == "sus2": return f"{root_name}sus2"
    if quality == "6": return f"{root_name}6"
    if quality == "m6": return f"{root_name}m6"
    if quality == "5": return f"{root_name}5"
    return f"{root_name}"

def chord_description_pt(symbol: str, quality: str) -> str:
    # Pequeno mapeamento para PT
    base = name_pt(symbol)  # converte tônica para PT
    mapa = {
        "M": "Maior",
        "m": "Menor",
        "7": "Sétima",
        "maj7": "Maior com Sétima Maior",
        "m7": "Menor com Sétima",
        "dim": "Diminuto",
        "dim7": "Diminuto com Sétima Diminuta",
        "m7b5": "Meio-diminuto (m7♭5)",
        "aug": "Aumentado",
        "sus4": "Suspenso na Quarta",
        "sus2": "Suspenso na Segunda",
        "6": "Maior com Sexta",
        "m6": "Menor com Sexta",
        "5": "Power chord",
    }
    return f"{base} {mapa.get(quality, '')}".strip()