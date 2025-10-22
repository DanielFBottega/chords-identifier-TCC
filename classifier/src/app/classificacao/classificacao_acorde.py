from dataclasses import dataclass
import numpy as np

@dataclass
class Acorde:
    nome: str
    confianca: float
    notas: list

NOTAS = ['C', 'C#', 'D', 'D#', 'E', 'F',
         'F#', 'G', 'G#', 'A', 'A#', 'B']

PADROES = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7]
}


def notas_para_indices(notas):
    """Remove oitavas e converte para índices 0–11."""
    base_notes = list(set([n[:-1].replace('♯', '#').replace('♭', 'b') for n in notas]))
    indices = []
    for n in base_notes:
        if n in NOTAS:
            indices.append(NOTAS.index(n))
        elif "b" in n:  # nota bemol
            i = (NOTAS.index(n.replace("b", "#")) - 1) % 12 if n.replace("b", "#") in NOTAS else None
            if i is not None:
                indices.append(i)
    return sorted(set(indices))


def diferencas_intervalares(indices, raiz):
    """Retorna os intervalos relativos à nota raiz."""
    return sorted([(i - raiz) % 12 for i in indices])


def calcular_confianca(intervalos, shape):
    """Calcula confiança baseada em interseção e notas extras."""
    intersec = len(set(intervalos) & set(shape))
    extras = len(set(intervalos) - set(shape))
    total = len(shape)
    score = (intersec / total) - (extras * 0.15)
    return max(score, 0.0)


def classify_chord_from_notes(notas_detectadas):
    if len(notas_detectadas) < 2:
        return Acorde(nome="?", confianca=0.0, notas=notas_detectadas)

    idxs = notas_para_indices(notas_detectadas)
    if not idxs:
        return Acorde(nome="?", confianca=0.0, notas=notas_detectadas)

    melhores = []
    for raiz in idxs:
        intervalos = diferencas_intervalares(idxs, raiz)
        for nome, shape in PADROES.items():
            score = calcular_confianca(intervalos, shape)
            # bônus se os intervalos incluírem terça e quinta (base do acorde)
            if 4 in intervalos or 3 in intervalos:
                score += 0.1
            if 7 in intervalos:
                score += 0.05
            melhores.append((score, NOTAS[raiz], nome))

    melhores.sort(reverse=True, key=lambda x: x[0])
    top = melhores[0]

    confianca = round(min(top[0], 1.0), 2)
    nome_final = f"{top[1]}{top[2]}"
    return Acorde(nome=nome_final, confianca=confianca, notas=notas_detectadas)
