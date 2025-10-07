
from typing import Dict, Any, Optional
import numpy as np
import librosa

def load_audio(path: str, sr: Optional[int] = None) -> Dict[str, Any]:
    y, sr_ret = librosa.load(path, sr=sr, mono=True)
    duration = len(y) / float(sr_ret)
    return {"y": y, "sr": sr_ret, "duration": duration}

def process_file(path: str, sr: Optional[int] = None, n_fft: int = 2048,
                 hop_length: int = 512, bins_per_octave: int = 12, n_bins: int = 88) -> Dict[str, Any]:
    
    audio = load_audio(path, sr=sr)
    y = audio["y"]
    sr_ret = audio["sr"]

    #CQT
    C = librosa.cqt(
        y=y, 
        sr=sr_ret,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    )

    #magnitude spectrogram
    C_mag = np.abs(C)
    S_db = librosa.amplitude_to_db(C_mag, ref=np.max)
    freqs = librosa.cqt_frequencies(n_bins=n_bins,
                                    fmin=librosa.note_to_hz('C1'),
                                    bins_per_octave=bins_per_octave)
    #tempo
    times = librosa.frames_to_time(
        np.arange(
            S_db.shape[1]),
        sr=sr_ret, 
        hop_length=hop_length)

    # placeholder chords detection: returns empty lists (controller will apply more robust method)
    chords = {"labels": [], "times": times, "confidences": []}

    return {"audio": {"sr": sr_ret, "duration": audio["duration"]},
            "spectrogram": {"S_db": S_db, "times": times, "freqs": freqs},
            "chords": chords}
