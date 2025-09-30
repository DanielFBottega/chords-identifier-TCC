
from typing import Dict, Any, Optional
import numpy as np
import librosa

def load_audio(path: str, sr: Optional[int] = None) -> Dict[str, Any]:
    y, sr_ret = librosa.load(path, sr=sr, mono=True)
    duration = len(y) / float(sr_ret)
    return {"y": y, "sr": sr_ret, "duration": duration}

def process_file(path: str, sr: Optional[int] = None, n_fft: int = 2048,
                 hop_length: int = 512, use_mel: bool = True, n_mels: int = 128):
    """Minimal pipeline: load + spectrogram + dummy chord detection placeholder."""
    audio = load_audio(path, sr=sr)
    y = audio["y"]
    sr_ret = audio["sr"]

    # spectrogram
    if use_mel:
        S = librosa.feature.melspectrogram(y=y, sr=sr_ret, n_fft=n_fft,
                                           hop_length=hop_length, n_mels=n_mels, power=2.0)
        S_db = librosa.power_to_db(S, ref=np.max)
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr_ret/2.0)
    else:
        S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))**2
        S_db = librosa.power_to_db(S, ref=np.max)
        freqs = np.linspace(0, sr_ret/2.0, S_db.shape[0])

    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr_ret, hop_length=hop_length)

    # placeholder chords detection: returns empty lists (controller will apply more robust method)
    chords = {"labels": [], "times": times, "confidences": []}

    return {"audio": {"sr": sr_ret, "duration": audio["duration"]},
            "spectrogram": {"S_db": S_db, "times": times, "freqs": freqs},
            "chords": chords}
