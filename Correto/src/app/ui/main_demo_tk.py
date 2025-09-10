import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io
import librosa

# Imports do projeto
from src.core.core_audio import process_file, AudioProcessingError, AudioLoadError
from src.utils.plotting import create_spectrogram_figure


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Identificação de Acordes")
        self.configure(bg="white")
        self.geometry("780x920")

        # Título
        tk.Label(
            self,
            text="Protótipo Para Identificação de Acordes Musicais",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#1976d2",
            pady=10
        ).pack(fill="x")

        # Botão carregar
        tk.Button(
            self, text="Carregar Arquivo de Áudio",
            font=("Segoe UI", 12, "bold"),
            bg="#2e7d32", fg="white",
            command=self._abrir_e_processar
        ).pack(pady=10)

        # Seções
        self._section_title("Espectrograma")
        self.lbl_spec_img = tk.Label(self, bg="white")
        self.lbl_spec_img.pack(padx=10, pady=5, fill="both")

        self._section_title("Notas Detectadas (CQT)")
        self.lbl_chroma_img = tk.Label(self, bg="white")
        self.lbl_chroma_img.pack(padx=10, pady=5, fill="both")

        self._section_title("Resultado")
        self.lbl_result = tk.Label(self, text="—", font=("Segoe UI", 14, "bold"),
                                   fg="#2e7d32", bg="white")
        self.lbl_result.pack(pady=10)

    def _section_title(self, text):
        tk.Label(self, text=text, font=("Segoe UI", 12, "bold"),
                 bg="white").pack(anchor="w", padx=10, pady=(10, 2))

    def _figure_to_label(self, fig, label_widget):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        plt.close(fig)
        buf.seek(0)
        im = Image.open(buf)
        tk_img = ImageTk.PhotoImage(im)
        label_widget.configure(image=tk_img)
        label_widget.image = tk_img

    def _abrir_e_processar(self):
        path = filedialog.askopenfilename(
            title="Selecione um arquivo de áudio",
            filetypes=[("Áudio", "*.wav *.mp3 *.flac *.ogg"), ("Todos", "*.*")]
        )
        if path:
            self._processar_arquivo(path)

    def _processar_arquivo(self, path):
        try:
            out = process_file(path)
        except (AudioProcessingError, AudioLoadError) as e:
            messagebox.showerror("Erro", str(e))
            return

        # Espectrograma
        spec = out["spectrogram"]
        fig_spec = create_spectrogram_figure(
            S_db=spec["S_db"], times=spec["times"], freqs=spec["freqs"],
            title="Espectrograma"
        )
        fig_spec.set_size_inches(6, 2.5)
        self._figure_to_label(fig_spec, self.lbl_spec_img)

        # Notas com oitava (CQT limitado ao range do piano)
        y, sr = librosa.load(path, sr=out["audio"]["sr"])

        fmin = librosa.note_to_hz('A0')   # menor nota do piano
        n_bins = 88                       # 88 teclas
        C = np.abs(librosa.cqt(
            y, sr=sr, hop_length=512,
            fmin=fmin, n_bins=n_bins, bins_per_octave=12
        ))

        C_mean = C.mean(axis=1)
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin)
        notas = [librosa.hz_to_note(f) for f in freqs]

        
        # 🔑 filtro: mantém apenas notas com intensidade > 0.1
        notas_filtradas = []
        intensidades_filtradas = []
        for n, v in zip(notas, C_mean):
            if v > 0.1:
                notas_filtradas.append(n)
                intensidades_filtradas.append(v)

        fig_ch, ax = plt.subplots(figsize=(7, 3))
        ax.bar(notas_filtradas, intensidades_filtradas)
        ax.set_xticks(range(len(notas_filtradas)))
        ax.set_xticklabels(notas_filtradas, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Intensidade")
        self._figure_to_label(fig_ch, self.lbl_chroma_img)

        # Resultado simples: tecla mais forte
        if intensidades_filtradas:
            idx_max = int(np.argmax(intensidades_filtradas))
            self.lbl_result.config(text=f"Tecla dominante: {notas_filtradas[idx_max]}")
        else:
            self.lbl_result.config(text="Nenhuma nota significativa detectada")


if __name__ == "__main__":
    app = App()
    app.mainloop()
