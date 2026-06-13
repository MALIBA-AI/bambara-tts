from typing import Optional

from maliba_ai.settings.tts.settings import InferenceOutput

# Supported TTS architectures (matching maliba_ai.settings.settings.Models).
SPARK_TTS = "spark_tts"
VITS = "vits"


class TTS:
    """
    Unified entry point for Text-To-Speech.

    Dispatches to the underlying model wrapper based on the chosen
    architecture while exposing a single `synthesize` interface.

    - spark_tts (BamSparkTTS): Bambara, speaker-conditioned via `speaker_id`.
    - vits (MalianTTS): multilingual Malian TTS, selected via `language`.

    Example:
        tts = TTS()                                  # spark_tts
        tts.synthesize(text, speaker_id=Speakers.Adama)

        tts = TTS(architecture="vits")
        tts.synthesize(text, language="bambara")
    """

    def __init__(
        self,
        architecture: str = SPARK_TTS,
        model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.architecture = architecture

        if architecture in (SPARK_TTS, "bam_spark"):
            from maliba_ai.core.tts.models.sparktts.inference import BamSparkTTS

            self._model = (
                BamSparkTTS(model_path, **kwargs)
                if model_path
                else BamSparkTTS(**kwargs)
            )

        elif architecture in (VITS, "malian_tts"):
            from maliba_ai.core.tts.models.vits.inference import MalianTTS

            self._model = (
                MalianTTS(model_path, **kwargs)
                if model_path
                else MalianTTS(**kwargs)
            )

        else:
            raise ValueError(
                f"Unsupported TTS architecture: {architecture}. "
                f"Available: {SPARK_TTS}, {VITS}"
            )

    def synthesize(self, text: str, **kwargs) -> InferenceOutput:
        """Generate speech audio from text. Returns an InferenceOutput."""
        return self._model.synthesize(text, **kwargs)
