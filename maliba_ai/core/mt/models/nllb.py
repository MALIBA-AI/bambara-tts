from typing import List, Union

import torch
from transformers import pipeline

from maliba_ai.settings.mt.nllb import Language, Settings
from maliba_ai.settings.mt.settings import TranslationOutput


class Translator:
    def __init__(self, model_name: str = Settings.model_repo, max_length: int = 512):
        """
        Initialize the Bambara Translator.

        Args:
            model_name (str, optional): Hugging Face model ID or local path.
            max_length (int, optional): Maximum sequence length. Defaults to 512.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._translator = pipeline(
            task="translation",
            model=model_name,
            device=0 if self._device.type == "cuda" else -1,
            max_length=max_length,
            truncation=True,
        )

    def translate(
        self,
        text: Union[str, List[str]],
        src_lang: Language,
        tgt_lang: Language,
        num_beams: int = 2,
    ) -> TranslationOutput:
        """
        Translate text from source language to target language.

        Args:
            text (Union[str, List[str]]): Input text or list of texts.
            src_lang (Language): Source language.
            tgt_lang (Language): Target language.
            num_beams (int, optional): Beam search parameter. Defaults to 2.

        Returns:
            TranslationOutput: Object containing translation result or error.
        """
        if not text:
            return TranslationOutput(error_message="text cannot be empty")

        if src_lang == tgt_lang:
            return TranslationOutput(
                translation=None,
                error_message="source and target languages must be different",
            )

        try:
            translation = self._translator(
                text,
                src_lang=src_lang.code,
                tgt_lang=tgt_lang.code,
                num_beams=num_beams,
            )
            return TranslationOutput(
                translation=str(translation[0]["translation_text"])
            )

        except Exception as e:
            return TranslationOutput(error_message=f"error during translation: {e}")


# if __name__ == "__main__":
#     translator = Translator()
#     result = translator.translate(
#         text="Hello my name is Moussa, I'm a student in Bamako, Mali. I love programming and artificial intelligence.",
#         src_lang=Languages.english,
#         tgt_lang=Languages.bambara
#     )
#     print(result)
