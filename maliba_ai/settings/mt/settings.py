
from typing import Optional, Union, List
from dataclasses import dataclass

@dataclass
class TranslationOutput:
    translation: Optional[Union[str, List[str]]] = None
    error_message: Optional[str] = None



