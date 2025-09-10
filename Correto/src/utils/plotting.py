from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def create_spectrogram_figure(S_db: np.ndarray,
                              times: np.ndarray,
                              freqs: np.ndarray,
                              title: str = "Espectrograma",
                              facecolor: str = "#ffffff") -> Figure:
    """
    Cria Figure do Matplotlib para o espectrograma.
    facecolor deve estar em HEX (evita 'SystemButtonFace').
    """
    fig = Figure(figsize=(7, 4), facecolor=facecolor)
    ax = fig.add_subplot(111)
    img = ax.imshow(S_db, aspect='auto', origin='lower',
                    extent=[times.min(), times.max(), freqs.min(), freqs.max()],
                    cmap='magma')
    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Frequência (Hz)")
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def create_chord_timeline_figure(labels, times,
                                 title: str = "Acordes por tempo",
                                 facecolor: str = "#ffffff") -> Figure:
    """
    Cria Figure simples com bars categóricos por intervalo de tempo.
    """
    if len(labels) == 0:
        labels = ["—"]
        times = [0.0]

    # Converte frames para segmentos
    # cada label vale de times[i] até times[i+1]
    starts = times[:-1]
    ends = times[1:]
    if len(starts) == 0:
        starts = [0.0]
        ends = [max(0.01, float(times[-1]) if len(times) > 0 else 0.5)]
    durations = [e - s for s, e in zip(starts, ends)]
    cats = labels[:-1] if len(labels) > 1 else labels

    fig = Figure(figsize=(7, 2.8), facecolor=facecolor)
    ax = fig.add_subplot(111)
    # plota blocos horizontais
    y = 0
    left = 0.0
    for i, dur in enumerate(durations):
        ax.barh(y, width=dur, left=starts[i], height=0.6, align='center',
                color="#2a9d8f", edgecolor="#1b5e55")
    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_yticks([])
    fig.tight_layout()
    return fig