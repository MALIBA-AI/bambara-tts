from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class TranslationOutput:
    translation: Optional[Union[str, List[str]]] = None
    error_message: Optional[str] = None
