from __future__ import annotations

import numpy as np
import librosa
import librosa.feature
import soundfile as sf

from typing import Dict, Any, Optional, Tuple, List
from .exceptions import AudioLoadError, AudioProcessingError

# Mapeamento de notas para índices de cromas (C=0, C#=1, ..., B=11)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

def load_audio(path: str, sr: Optional[int] = None, mono: bool = True) -> Dict[str, Any]:
    """
    Carrega um arquivo de áudio usando librosa.
    Retorna um dicionário com: y (np.ndarray), sr (int), duration (float).
    Lança AudioLoadError em caso de falha.
    """
    try:
        y, sr_actual = librosa.load(path, sr=sr, mono=mono)
        duration = float(len(y) / sr_actual) if y.size > 0 else 0.0
        return {"y": y, "sr": sr_actual, "duration": duration}
    except Exception as e:
        raise AudioLoadError(f"Falha ao carregar áudio '{path}': {e}") from e


def compute_spectrogram(y: np.ndarray,
                        sr: int,
                        n_fft: int = 2048,
                        hop_length: int = 512,
                        use_mel: bool = True,
                        n_mels: int = 128) -> Dict[str, Any]:
    """
    Calcula espectrograma em dB (mel ou STFT). Retorna Dict com S_db, times, freqs.
    """
    try:
        if use_mel:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                               hop_length=hop_length, n_mels=n_mels, power=2.0)
            S_db = librosa.power_to_db(S, ref=np.max)
            freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2.0)
        else:
            S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2
            S_db = librosa.power_to_db(S, ref=np.max)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)

        return {
            "S_db": S_db,
            "times": times,
            "freqs": freqs
        }
    except Exception as e:
        raise AudioProcessingError(f"Erro ao computar espectrograma: {e}") from e


def _triad_templates() -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna templates (12x12) para acordes maiores e menores baseados em cromas.
    major_templates[root, pitch] = 1 em {root, root+4, root+7}
    minor_templates[root, pitch] = 1 em {root, root+3, root+7}
    """
    major = np.zeros((12, 12), dtype=float)
    minor = np.zeros((12, 12), dtype=float)
    for r in range(12):
        major[r, r] = 1.0
        major[r, (r + 4) % 12] = 1.0
        major[r, (r + 7) % 12] = 1.0

        minor[r, r] = 1.0
        minor[r, (r + 3) % 12] = 1.0
        minor[r, (r + 7) % 12] = 1.0

    # normaliza por norma L2 para usar produto interno como similaridade
    major /= np.linalg.norm(major, axis=1, keepdims=True)
    minor /= np.linalg.norm(minor, axis=1, keepdims=True)
    return major, minor


def _choose_seventh(chroma_vec: np.ndarray, root: int, base_kind: str,
                    seventh_threshold: float = 0.35) -> Optional[str]:
    """
    Decide se adiciona 7ª: '7' (dominante, b7) ou 'maj7' (7M).
    Usa a energia relativa nos graus b7 (root+10) e 7M (root+11).
    Retorna '7', 'maj7' ou None.
    """
    # energia relativa dos graus
    e_b7 = chroma_vec[(root + 10) % 12]
    e_7M = chroma_vec[(root + 11) % 12]
    # normaliza pelo max para robustez
    max_e = float(chroma_vec.max() + 1e-9)
    e_b7_rel = e_b7 / max_e
    e_7M_rel = e_7M / max_e

    if e_b7_rel >= seventh_threshold and e_b7_rel >= e_7M_rel * 1.05:
        return "7"
    if e_7M_rel >= seventh_threshold and e_7M_rel >= e_b7_rel * 1.05:
        return "maj7"
    return None


def detect_chords(y: np.ndarray,
                  sr: int,
                  hop_length: int = 512,
                  tuning: Optional[float] = None,
                  smooth_window: int = 5) -> Dict[str, Any]:
    """
    Detecção simples de acordes por cromas + template matching (maior/menor) e heurística de 7ª.

    Retorna:
    {
      "labels": List[str],  # rótulo por frame
      "times": np.ndarray,  # tempo por frame (s)
      "confidences": List[float],  # similaridade [0..1]
      "frame_hop_s": float
    }
    """
    try:
        # Croma por CQT é mais robusto para harmonia
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, tuning=tuning)
        times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

        # normaliza vetores por coluna (frame) para similaridade consistente
        chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)

        major_templates, minor_templates = _triad_templates()

        labels: List[str] = []
        confidences: List[float] = []

        for i in range(chroma_norm.shape[1]):
            v = chroma_norm[:, i]
            # similaridade com todas as tríades
            sim_major = major_templates @ v
            sim_minor = minor_templates @ v

            best_major_idx = int(np.argmax(sim_major))
            best_minor_idx = int(np.argmax(sim_minor))
            best_major = float(sim_major[best_major_idx])
            best_minor = float(sim_minor[best_minor_idx])

            if best_major >= best_minor:
                root = best_major_idx
                kind = "maj"
                conf = best_major
            else:
                root = best_minor_idx
                kind = "min"
                conf = best_minor

            # heurística de 7ª
            seventh = _choose_seventh(chroma[:, i], root, kind)
            name = NOTE_NAMES[root]
            if kind == "min":
                name += "m"
            if seventh == "7":
                name += "7"
            elif seventh == "maj7":
                name += "maj7"

            labels.append(name)
            confidences.append(float(np.clip(conf, 0.0, 1.0)))

        # suavização temporal (modo mediana deslizante sobre labels)
        if smooth_window > 1 and len(labels) >= smooth_window:
            labels = _median_smooth_labels(labels, window=smooth_window)

        return {
            "labels": labels,
            "times": times,
            "confidences": confidences,
            "frame_hop_s": hop_length / float(sr)
        }
    except Exception as e:
        raise AudioProcessingError(f"Erro na detecção de acordes: {e}") from e


def _median_smooth_labels(labels: List[str], window: int = 5) -> List[str]:
    """
    Suaviza labels discretas por janela deslizante (moda).
    """
    from collections import Counter
    half = window // 2
    n = len(labels)
    out = labels.copy()
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        segment = labels[a:b]
        c = Counter(segment)
        out[i] = c.most_common(1)[0][0]
    return out


def process_file(path: str,
                 sr: Optional[int] = None,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 use_mel: bool = True,
                 n_mels: int = 128,
                 tuning: Optional[float] = None,
                 smooth_window: int = 5) -> Dict[str, Any]:
    """
    Pipeline único para arquivo:
    - carrega
    - espectrograma
    - detecção de acordes

    Retorna dict padronizado para evitar desempacotamento incorreto.
    """
    audio = load_audio(path, sr=sr)
    spec = compute_spectrogram(audio["y"], audio["sr"], n_fft=n_fft,
                               hop_length=hop_length, use_mel=use_mel, n_mels=n_mels)
    chords = detect_chords(audio["y"], audio["sr"], hop_length=hop_length,
                           tuning=tuning, smooth_window=smooth_window)
    return {
        "audio": {
            "sr": audio["sr"],
            "duration": audio["duration"]
        },
        "spectrogram": spec,
        "chords": chords
    }


def save_wav(path: str, y: np.ndarray, sr: int) -> None:
    """Utilitário simples para salvar WAV."""
    sf.write(path, y, sr)