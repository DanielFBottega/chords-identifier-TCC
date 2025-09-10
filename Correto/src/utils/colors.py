def tk_color_to_hex(widget) -> str:
    """
    Converte a cor do Tkinter (ex.: 'SystemButtonFace') para HEX (#RRGGBB)
    para ser aceita pelo Matplotlib. Evita o erro "Invalid RGBA argument: 'SystemButtonFace'".
    """
    try:
        # tk.winfo_rgb retorna tupla de (r, g, b) em 16 bits (0..65535).
        r16, g16, b16 = widget.winfo_rgb(widget.cget("background"))
        r = int(r16 / 65535 * 255)
        g = int(g16 / 65535 * 255)
        b = int(b16 / 65535 * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        # fallback para branco
        return "#ffffff"