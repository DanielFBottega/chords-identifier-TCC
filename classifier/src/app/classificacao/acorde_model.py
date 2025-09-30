
from dataclasses import dataclass
from typing import List

@dataclass
class AcordeModel:
    name: str
    pcs: List[int]  # pitch classes 0..11 for the chord template
