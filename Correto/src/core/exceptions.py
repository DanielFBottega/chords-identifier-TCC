class AudioProcessingError(Exception):
    """Erro genérico de processamento de áudio."""
    pass


class AudioLoadError(AudioProcessingError):
    """Erro ao carregar arquivo de áudio."""
    pass