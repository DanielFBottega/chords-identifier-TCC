import numpy as np
import librosa
import matplotlib.pyplot as plt

def create_chroma_figure(y, sr, hop_length=512):
    """
    Cria um gráfico de barras representando o vetor chroma médio do áudio.
    """
    # Calcula chroma CQT (média temporal)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)

    notas = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Gera figura de barras
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.bar(range(12), chroma_mean, color="#1976d2")
    ax.set_xticks(range(12))
    ax.set_xticklabels(notas)
    ax.set_ylim(0, 1)
    ax.set_title("Vetor Chroma Médio", fontsize=10)
    plt.tight_layout()

    return fig, chroma_mean, notas
