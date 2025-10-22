from typing import Dict, Any
import numpy as np
import librosa
from scipy.signal import find_peaks

def process_file(path: str) -> Dict[str, Any]:
    """
    Processa o arquivo de áudio e retorna dados de espectrograma e picos harmônicos.
    Mantém compatibilidade com a interface esperada pelo controlador_principal.
    """
    y, sr = librosa.load(path, sr=None, mono=True)

    hop_length = 512
    bins_per_octave = 12
    n_bins = 88
    fmin = librosa.note_to_hz("C1")

    # CQT
    C = np.abs(librosa.cqt(
        y=y, sr=sr, hop_length=hop_length,
        fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave
    ))

    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop_length)

    mean_energy = np.mean(C)
    C_thresh = np.where(C > mean_energy * 1.8, C, 0)

    mean_spectrum = np.mean(C_thresh, axis=1)
    peaks, _ = find_peaks(mean_spectrum, height=np.max(mean_spectrum) * 0.25)

    # Notas + oitava
    print(peaks)
    detected_notes = [librosa.hz_to_note(freqs[p]) for p in peaks]

    # Retorno esperado
    return {
        "spectrogram": {
            "S_db": librosa.amplitude_to_db(np.abs(C), ref=np.max),
            "times": times,
            "freqs": freqs
        },
        "harmonic_peaks": peaks,
        "detected_notes": detected_notes,
        "C": C,
        "sr": sr,
        "y": y
    }
