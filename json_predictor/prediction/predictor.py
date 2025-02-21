from abc import ABC, abstractmethod
from typing import Tuple
"""
general predictor output
"""
TokenInfo = Tuple[str, float, float]    # next str next_word, float softmax, float semantic similarity score
"""
This is the abstract class definition for the general predictor class
"""

class Predictor(ABC):
    @abstractmethod
    def next_token(input_text: str) -> TokenInfo:
        ...
