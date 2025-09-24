
from dataclasses import dataclass
from typing import Optional

@dataclass
class Language:
    code: str
class Languages:
    bambara = Language(code="bam_Latn")
    french =  Language(code="fra_Latn")
    english = Language(code="eng_Latn")

class Settings:
    model_repo:str = "sudoping01/nllb-bambara-v2"
    Languages = Languages




