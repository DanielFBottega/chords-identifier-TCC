import os
import numpy as np
from src.core.core_audio import compute_spectrogram, detect_chords, save_wav

def test_compute_spectrogram():
    # Sinal simples (seno)
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    spec = compute_spectrogram(y, sr)
    assert "S_db" in spec and "times" in spec and "freqs" in spec
    assert spec["S_db"].ndim == 2

def test_detect_chords():
    # Acorde C (C-E-G)
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.4 * np.sin(2 * np.pi * 261.63 * t) +
         0.4 * np.sin(2 * np.pi * 329.63 * t) +
         0.4 * np.sin(2 * np.pi * 392.00 * t)).astype(np.float32)

    chords = detect_chords(y, sr, hop_length=512, smooth_window=3)
    assert "labels" in chords and len(chords["labels"]) > 0
    # espera-se que C ou Cm apareça majoritariamente como C
    # não é um teste estrito, apenas sanidade
    unique = set(chords["labels"])
    assert any(lbl.startswith("C") for lbl in unique)