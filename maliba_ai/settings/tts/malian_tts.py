from typing import List


class Submodels:
    bambara = "bambara"
    boomu = "boomu"
    dogon = "dogon"
    pular = "pular"
    songhoy = "songhoy"
    tamasheq = "tamasheq"
    all_languages = [bambara, boomu, dogon, pular, songhoy, tamasheq]


class Settings:
    models_repo: str = "MALIBA-AI/malian-tts"
    models: List[str] = Submodels.all_languages
    models_subfolder: str = "models"
