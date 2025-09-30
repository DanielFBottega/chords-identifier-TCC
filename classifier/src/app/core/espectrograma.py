
from matplotlib.figure import Figure
import numpy as np

def create_spectrogram_figure(S_db, times, freqs, title="Espectrograma", facecolor="#ffffff"):
    fig = Figure(figsize=(6, 2.5), facecolor=facecolor)
    ax = fig.add_subplot(111)
    im = ax.imshow(S_db, aspect="auto", origin="lower", cmap="magma", extent=[times.min() if len(times)>0 else 0, times.max() if len(times)>0 else 1, freqs.min() if len(freqs)>0 else 0, freqs.max() if len(freqs)>0 else 1])
    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_yticks([])
    fig.tight_layout()
    return fig
