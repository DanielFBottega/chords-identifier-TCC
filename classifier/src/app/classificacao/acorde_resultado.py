
from dataclasses import dataclass
from typing import List

@dataclass
class AcordeResultado:
    nome: str
    confianca: float
    notas: List[str]
