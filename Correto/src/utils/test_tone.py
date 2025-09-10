"""
Gera um arquivo WAV com um acorde de C maior (C-E-G) para teste rápido.
Uso: python -m src.utils.test_tone
"""
import numpy as np
import soundfile as sf
import os

def synth_chord(sr=22050, seconds=2.0, freqs=(261.63, 329.63, 392.00)):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = sum([0.33 * np.sin(2 * np.pi * f * t) for f in freqs])
    # leve fade para evitar clicks
    fade_len = int(0.01 * sr)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    y[:fade_len] *= fade_in
    y[-fade_len:] *= fade_out
    return y.astype(np.float32), sr

if __name__ == "__main__":
    y, sr = synth_chord()
    out = os.path.join(".", "tone_Cmaj.wav")
    sf.write(out, y, sr)
    print(f"Arquivo gerado: {out}")