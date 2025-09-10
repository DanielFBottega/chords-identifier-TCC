import numpy as np
import librosa
from typing import List, Dict
from .types import NoteEvent
from .note_utils import midi_to_name, hz_to_midi

class NoteTracker:
    def __init__(self,
                 sr: int,
                 hop_length: int = 512,
                 fmin: str = "C2",
                 n_bins: int = 72,
                 bins_per_octave: int = 12,
                 peak_top_n: int = 5,
                 db_threshold: float = -48.0,
                 min_note_gap: float = 0.08):
        self.sr = sr
        self.hop_length = hop_length
        self.peak_top_n = peak_top_n
        self.db_threshold = db_threshold
        self.min_note_gap = min_note_gap  # segundos mínimos entre dois "note-on" iguais

        self.C = None
        self.C_db = None
        self.freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=librosa.note_to_hz(fmin), bins_per_octave=bins_per_octave)
        self.min_gap_per_midi: Dict[int, float] = {}

    def compute_cqt(self, y):
        C = np.abs(librosa.cqt(y, sr=self.sr, hop_length=self.hop_length,
                               fmin=self.freqs[0], bins_per_octave=12,
                               n_bins=len(self.freqs)))
        self.C = C
        self.C_db = librosa.amplitude_to_db(C, ref=np.max)
        return self.C_db

    def detect_note_events(self) -> List[NoteEvent]:
        assert self.C_db is not None
        T = self.C_db.shape[1]
        events: List[NoteEvent] = []

        for t in range(T):
            frame_db = self.C_db[:, t]
            if np.all(np.isneginf(frame_db)):
                continue
            # filtra por limiar
            mask = frame_db > self.db_threshold
            if not np.any(mask):
                continue

            # pega top-N picos simples
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            # Ordena por energia
            top_idx = np.argsort(frame_db[idx])[-self.peak_top_n:]
            bins = idx[top_idx]

            time_sec = t * (self.hop_length / self.sr)
            for b in bins:
                f = float(self.freqs[b])
                midi_f = hz_to_midi(f)
                midi_i = int(round(midi_f))
                last_time = self.min_gap_per_midi.get(midi_i, -1e9)
                if time_sec - last_time < self.min_note_gap:
                    continue  # debouncing
                vel = float((frame_db[b] - self.db_threshold) / max(1e-6, (0 - self.db_threshold)))
                vel = float(np.clip(vel, 0.0, 1.0))
                events.append(NoteEvent(
                    time=time_sec,
                    midi=midi_i,
                    name=midi_to_name(midi_i),
                    freq=f,
                    velocity=vel,
                    source="cqt"
                ))
                self.min_gap_per_midi[midi_i] = time_sec

        # ordenar cronologicamente
        events.sort(key=lambda e: e.time)
        return events