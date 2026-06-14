import re
import numpy as np
import torch
import soundfile as sf

from typing import Optional

from maliba_ai.models.models import load_tts_model, load_audio_tokenizer
from maliba_ai.config.speakers import Adama, SingleSpeaker, Settings
from maliba_ai.utils.utils import normalize_text

_SEMANTIC_RE = re.compile(r"<\|bicodec_semantic_(\d+)\|>")
_GLOBAL_RE = re.compile(r"<\|bicodec_global_(\d+)\|>")


def build_tts_prompt(text: str) -> str:
    """Build the Spark-TTS prompt that the LLM completes into audio tokens."""
    return (
        "<|task_tts|>"
        "<|start_content|>"
        f"{text}"
        "<|end_content|>"
        "<|start_global_token|>"
    )


def parse_audio_tokens(generated_text: str):
    """Extract (global_ids, semantic_ids) from generated text."""
    semantic_ids = [int(t) for t in _SEMANTIC_RE.findall(generated_text)]
    global_ids = [int(t) for t in _GLOBAL_RE.findall(generated_text)]
    return global_ids, semantic_ids


class BambaraTTSInference:
    def __init__(self, model_path: Optional[str] = Settings.model_repo, max_seq_length: Optional[int] = 2048):
        """
        Initialize the Bambara TTS inference class.

        Args:
            model_path (str, optional): Path to the model (local or huggingface repo id).
            max_seq_length (int, optional): Maximum sequence length for generation.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._tokenizer = load_tts_model(model_path=model_path, max_seq_length=max_seq_length)
        self._audio_tokenizer = load_audio_tokenizer(self._device)

    @property
    def sample_rate(self) -> int:
        return self._audio_tokenizer.config.get("sample_rate", Settings.sample_rate)

    @torch.inference_mode()
    def _generate_speech_from_text(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_audio_tokens: int = 2048,
    ) -> np.ndarray:
        """
        Generate speech from pre-formatted text.

        Args:
            text (str): Pre-formatted text (with speaker ID if applicable).
            temperature (float): Sampling temperature (default: 0.8).
            top_k (int): Top-k sampling parameter (default: 50).
            top_p (float): Top-p sampling parameter (default: 1.0).
            max_new_audio_tokens (int): Maximum audio tokens to generate (default: 2048).

        Returns:
            np.ndarray: Generated waveform as a NumPy array.
        """
        prompt = build_tts_prompt(text)
        model_inputs = self._tokenizer([prompt], return_tensors="pt").to(self._device)

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=max_new_audio_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
        predicted_text = self._tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]

        global_ids, semantic_ids = parse_audio_tokens(predicted_text)

        return self._audio_tokenizer.wav_from_token_ids(global_ids, semantic_ids)

    def generate_speech(
        self,
        text: str,
        speaker_id: Optional[SingleSpeaker] = Adama,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_audio_tokens: int = 2048,
        output_filename: str = None,
    ) -> np.ndarray:
        """
        Generate speech from text with optional speaker ID.

        Args:
            text (str): Input text in Bambara to convert to speech.
            speaker_id (SingleSpeaker, optional): Speaker (e.g. Speakers.Adama).
            temperature (float): Sampling temperature (default: 0.8).
            top_k (int): Top-k sampling parameter (default: 50).
            top_p (float): Top-p sampling parameter (default: 1.0).
            max_new_audio_tokens (int): Maximum audio tokens to generate (default: 2048).
            output_filename (str, optional): Name of output audio file.

        Returns:
            np.ndarray: Generated waveform as a NumPy array.
        """
        if not isinstance(text, str):
            raise TypeError("text should be a string")
        if not text.strip():
            raise ValueError("text can not be empty")
        if speaker_id is not None and speaker_id.id.upper() not in Settings.speakers_ids:
            raise ValueError(f"Speaker '{speaker_id.id}' is not supported")

        # Normalize the input the same way the model was trained on.
        text = normalize_text(text)

        formatted_text = f"{speaker_id.id}: {text}" if speaker_id else text
        generated_waveform = self._generate_speech_from_text(
            text=formatted_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_audio_tokens=max_new_audio_tokens,
        )

        if generated_waveform.size > 0 and output_filename:
            sf.write(output_filename, generated_waveform, self.sample_rate)

        return generated_waveform
