from typing import Dict, Any
from src.app.core.processar_audio import process_file
from src.app.classificacao.classificacao_acorde import classify_chord_from_notes
from src.app.core.vetor_chroma import create_chroma_figure
import numpy as np
import librosa

class ControladorPrincipal:
    def __init__(self):
        pass

    def processar_arquivo_audio(self, path: str) -> Dict[str, Any]:
        # Extrai dados espectrais
        data = process_file(path)
        y, sr = data["y"], data["sr"]

        # Calcula vetor chroma (para visualização)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_norm = chroma_mean / np.sum(chroma_mean)
        pc_vector = np.where(chroma_norm > 0.1, chroma_norm, 0.0)
        if np.sum(pc_vector) > 0:
            pc_vector = librosa.util.normalize(pc_vector, norm=1)

        # Figura do chroma
        fig_chroma, _, notas_fig = create_chroma_figure(y, sr)

        # Classificação harmônica
        print(data['detected_notes'])
        chord_result = classify_chord_from_notes(data["detected_notes"])

        # Monta resultado final compatível com o main_demo
        return {
            "spectrogram": data["spectrogram"],
            "harmonic_peaks": data["harmonic_peaks"],
            "detected_notes": chord_result.notas,
            "chord_result": chord_result,
            "_pc_vector": pc_vector.tolist(),
            "chroma_figure": fig_chroma
        }
