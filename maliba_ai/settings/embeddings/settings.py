from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class EmbeddingOutput:
    vector: Optional[np.ndarray] = None
    error_message: Optional[str] = None


@dataclass
class SimilarWordsOutput:
    words: Optional[List[Tuple[str, float]]] = None
    error_message: Optional[str] = None


@dataclass
class SimilarityOutput:
    score: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class TextSearchOutput:
    results: Optional[List[Tuple[float, str, int]]] = None
    error_message: Optional[str] = None
