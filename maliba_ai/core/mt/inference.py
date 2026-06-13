from typing import List, Optional, Union

from maliba_ai.settings.mt.nllb import Language, Languages
from maliba_ai.settings.mt.settings import TranslationOutput

# Supported MT architectures (matching maliba_ai.settings.settings.Models).
NLLB = "nllb"


class MachineTranslation:
    """
    Unified entry point for Machine Translation.

    Dispatches to the underlying model wrapper based on the chosen
    architecture while exposing a single `translate` interface.

    Example:
        mt = MachineTranslation()
        out = mt.translate("Bonjour", Languages.french, Languages.bambara)
    """

    languages = Languages

    def __init__(
        self,
        architecture: str = NLLB,
        model_name: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        self.architecture = architecture

        if architecture == NLLB:
            from maliba_ai.core.mt.models.nllb import BambaraTranslator
            from maliba_ai.settings.mt.nllb import Settings

            self._model = BambaraTranslator(
                model_name or Settings.model_repo, max_length
            )
        else:
            raise ValueError(
                f"Unsupported MT architecture: {architecture}. Available: {NLLB}"
            )

    def translate(
        self,
        text: Union[str, List[str]],
        src_lang: Language,
        tgt_lang: Language,
        num_beams: int = 2,
    ) -> TranslationOutput:
        """Translate text from a source language to a target language."""
        return self._model.translate(text, src_lang, tgt_lang, num_beams)
