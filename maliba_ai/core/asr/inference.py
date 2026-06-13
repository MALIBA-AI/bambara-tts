from typing import Optional, Union

from maliba_ai.settings.asr.settings import Settings

# Supported ASR architectures (matching maliba_ai.settings.settings.Models).
WHISPER = "whisper"
GEMMA = "gemma-3n"


class SpeechRecognition:
    """
    Unified entry point for Automatic Speech Recognition.

    Dispatches to the underlying model wrapper based on the chosen
    architecture while exposing a single `transcribe` interface.

    Example:
        asr = SpeechRecognition()                      # whisper, default repo
        asr = SpeechRecognition(architecture="gemma-3n")
        result = asr.transcribe("audio.wav")           # -> {"text": ...}
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        architecture: str = WHISPER,
    ) -> None:
        self.architecture = architecture

        if architecture == WHISPER:
            from maliba_ai.core.asr.models.whisper_base import ASR

            self._model = ASR(model_id or Settings.whisper_model_repo)

        elif architecture in (GEMMA, "gemma"):
            from maliba_ai.core.asr.models.gemma_base import GemmaASR

            self._model = GemmaASR(model_id or Settings.gemma_model_repo)

        else:
            raise ValueError(
                f"Unsupported ASR architecture: {architecture}. "
                f"Available: {WHISPER}, {GEMMA}"
            )

    def transcribe(self, audio: Union[str, bytes], **kwargs) -> dict:
        """Transcribe audio into text. Returns a dict of shape {"text": ...}."""
        return self._model.transcribe(audio, **kwargs)
