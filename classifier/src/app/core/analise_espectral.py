
from .vetor_chroma import chroma_cqt_mean

def extrair_vetor_chroma(y, sr):
    notes, vals = chroma_cqt_mean(y, sr)
    return {"notes": notes, "values": vals}
