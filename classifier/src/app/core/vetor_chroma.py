
import numpy as np
import librosa

def chroma_cqt_mean(y, sr, hop_length=512, fmin=None, n_bins=88):
    if fmin is None:
        fmin = librosa.note_to_hz('A0')
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=12))
    C_mean = C.mean(axis=1)
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin)
    notes = [librosa.hz_to_note(f) for f in freqs]
    return notes, C_mean
