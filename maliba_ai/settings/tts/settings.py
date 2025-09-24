from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class InferenceOutput:
    waveform: Optional[np.ndarray]
    sample_rate: Optional[int]
    error_message: Optional[str] = None
    output_filename: Optional[str] = None
    

    def is_successful(self) -> bool:
        return self.error_message is None and self.waveform is not None and self.sample_rate is not None
