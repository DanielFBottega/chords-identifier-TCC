# src/app/controllers/controlador_principal.py
from typing import Dict, Any
from src.app.core.processar_audio import process_file
from src.app.classificacao.classificacao_acorde import classify_chord_from_pitchclass_vector
import numpy as np
import librosa

class ControladorPrincipal:
    def __init__(self):
        pass

    def processar_arquivo_audio(self, path: str) -> Dict[str, Any]:
        out = process_file(path)
        # load audio at pipeline sr
        y, sr = librosa.load(path, sr=out["audio"]["sr"], mono=True)

        # Parameters
        hop_length = 512
        bins_per_octave = 12
        n_bins = 88

        #CQT
        C = librosa.cqt(
            y=y, 
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave
        )
        C_mag = np.abs(C)


        #Chroma
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
            n_chroma=12,
            bins_per_octave=bins_per_octave
        )

        # vetor médio
        pc_vector = np.mean(chroma, axis=1)

        # normalização
        pc_vector = pc_vector / (np.sum(pc_vector) + 1e-6)

        # threshold simples para limpar resíduos
        threshold = 0.05
        pc_vector = np.where(pc_vector > threshold, pc_vector, 0.0)

        # classificação
        chord_result = classify_chord_from_pitchclass_vector(pc_vector)

        #notas detectadas
        detected_notes = chord_result.notas


        out["detected_notes"] = detected_notes
        out["chord_result"] = chord_result
        out["_pc_vector"] = pc_vector.tolist()
        return out
