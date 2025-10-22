
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.app.controllers.controlador_principal import ControladorPrincipal
from src.app.core.espectrograma import create_spectrogram_figure

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Protótipo - Identificação de Acordes")
        self.geometry("780x720")
        self.configure(bg="white")

        tk.Label(self, text="Protótipo Para Identificação de Acordes Musicais", font=("Segoe UI", 14, "bold"), bg="#1976d2", fg="white").pack(fill="x")
        tk.Button(self, text="Carregar Arquivo de Áudio", bg="#2e7d32", fg="white", command=self._abrir).pack(pady=8)

        self._section("Espectrograma")
        self.img_spec = tk.Label(self, bg="white", bd=1, relief="solid")
        self.img_spec.pack(fill="both", padx=12, pady=6)

        self._section("Notas detectadas (CQT reduzido)")
        self.img_chroma = tk.Label(self, bg="white", bd=1, relief="solid")
        self.img_chroma.pack(fill="both", padx=12, pady=6)

        self._section("Resultado")
        self.lbl_result = tk.Label(self, text="—", font=("Segoe UI", 12, "bold"), fg="#2e7d32", bg="white")
        self.lbl_result.pack(pady=6)

        self.ctrl = ControladorPrincipal()

    def _section(self, title):
        tk.Label(self, text=title, bg="white", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(8,2))

    def _figure_to_label(self, fig, label_widget):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        photo = ImageTk.PhotoImage(img)
        label_widget.configure(image=photo)
        label_widget.image = photo
        plt.close(fig)

    def _abrir(self):
        path = filedialog.askopenfilename(filetypes=[("Áudio",".wav .mp3 .flac .ogg")])
        if not path:
            return
        self._abrir_com_caminho(path)

    def _abrir_com_caminho(self, path):
        if not path:
            return
        try:
            out = self.ctrl.processar_arquivo_audio(path)
        except Exception as e:
            messagebox.showerror("Erro", str(e))
            return

        spec = out["spectrogram"]
        fig_spec = create_spectrogram_figure(spec["S_db"], spec["times"], spec["freqs"])
        self._figure_to_label(fig_spec, self.img_spec)
        
        fig_chroma = out.get("chroma_figure")
        if fig_chroma:
            self._figure_to_label(fig_chroma, self.img_chroma)
        else:
            fig, ax = plt.subplots(figsize=(6,2.2))
            ax.text(0.5,0.5,"Nenhuma nota significativa detectada", ha="center", va="center")
            ax.axis("off")
            self._figure_to_label(fig, self.img_chroma)

        chord = out.get("chord_result")
        if chord:
            self.lbl_result.config(text=f"{chord.nome}  ({chord.confianca*100:.0f}%)")
        else:
            self.lbl_result.config(text="—")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
        ctrl = ControladorPrincipal()
        result = ctrl.processar_arquivo_audio(audio_file_path)
        print("Resultado da Classificação:", result["chord_result"].nome)
        print("Confiança:", result["chord_result"].confianca)
        print("Notas Detectadas:", result["chord_result"].notas)
    else:
        App().mainloop()

