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
        n_fft = 2048

        # Compute chroma per frame (STFT chroma is fine for per-frame)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)  # shape (12, frames)

        # Compute frame energy to weight frames
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        energy = S.mean(axis=0)  # energy per frame

        # Normalize chroma per frame (to reduce loudness bias)
        chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)

        # For each frame compute similarity to each pitch-class basis (we'll just accumulate chroma itself)
        # Aggregate pitch-class vector by weighted average across frames (weight = frame energy)
        weights = energy + 1e-8
        weights = weights / (np.sum(weights) + 1e-12)
        pc_vector = np.dot(chroma, weights)  # shape (12,) weighted mean chroma per pitch class

        # Optionally apply a small threshold to zero-out tiny residuals (reduces harm.)
        threshold = 0.08  # tuneable (0.05..0.12)
        pc_vector = np.where(pc_vector >= threshold, pc_vector, 0.0)

        # Also normalize PC vector for classification
        # (classification function will normalize internally)
        chord_result = classify_chord_from_pitchclass_vector(pc_vector)

        # Also derive note list (for UI display): choose pitch classes above threshold
        NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        detected_notes = [NOTE_NAMES[i] for i,v in enumerate(pc_vector) if v > 0]

        out["detected_notes"] = detected_notes
        out["chord_result"] = chord_result
        out["_pc_vector"] = pc_vector.tolist()
        return out
