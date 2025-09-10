# Projeto Áudio - Nutrivital

Base funcional do core de áudio + controlador, pronta para integrar com UI Tkinter.

## Como instalar

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

## Teste rápido (CLI)

python -m src.utils.test_tone  # gera um WAV de teste em ./tone_Cmaj.wav
python -m src.app.controllers.controlador_principal --arquivo ./tone_Cmaj.wav

## Demo Tk (opcional)

python -m src.app.ui.main_demo_tk