"""BiCodec vocoder wrapper for the inference server.

Loads only the *decoding* path of BiCodec (no wav2vec2), and turns the integer
token ids parsed from the LLM output into a 16 kHz waveform.
"""

import numpy as np
import torch

from maliba_ai.models.models import load_audio_tokenizer


class BiCodecVocoder:
    def __init__(self, device: str = "", model_dir: str = "Spark-TTS-0.5B"):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # detokenize_only=True => skip the heavy wav2vec2 feature extractor and
        # only download/load what is needed to synthesize audio.
        self._tokenizer = load_audio_tokenizer(
            self.device, detokenize_only=True, model_dir=model_dir
        )

    @property
    def sample_rate(self) -> int:
        return int(self._tokenizer.config.get("sample_rate", 16000))

    def tokens_to_wav(
        self, global_token_ids: "list[int]", semantic_token_ids: "list[int]"
    ) -> np.ndarray:
        """Decode global + semantic token ids into a float32 waveform."""
        wav = self._tokenizer.wav_from_token_ids(global_token_ids, semantic_token_ids)
        return np.asarray(wav, dtype=np.float32)
