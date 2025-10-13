import re
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from maliba_ai.core.tts.models.sparktts.utils.loader import (
    load_audio_tokenizer, load_tts_model)
from maliba_ai.settings.tts.bam_spark import Settings, SingleSpeaker
from maliba_ai.settings.tts.settings import InferenceOutput


class BamSparkTTS:
    def __init__(
        self,
        model_path: Optional[str] = Settings.model_repo,
        max_seq_length: Optional[int] = 2048,
    ):
        """
        Initialize the BamSpark TTS inference class.

        Args:
            model_path (str, optional): Path to the model (local or huggingface repo id).
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._tokenizer = load_tts_model(
            model_path=model_path, max_seq_length=max_seq_length
        )
        self._audio_tokenizer = load_audio_tokenizer(self._device)

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
        prompt = "".join(
            [
                "<|task_tts|>",
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
            ]
        )

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

        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1] :]
        predicted_text = self._tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False
        )[0]

        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicted_text)

        if not semantic_matches:
            return np.array([], dtype=np.float32)

        pred_semantic_ids = (
            torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
        )

        if not global_matches:
            pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
        else:
            pred_global_ids = (
                torch.tensor([int(token) for token in global_matches])
                .long()
                .unsqueeze(0)
            )

        pred_global_ids = pred_global_ids.unsqueeze(0)  # Shape: (1, 1, N_global)

        self._audio_tokenizer.device = self._device
        self._audio_tokenizer.model.to(self._device)

        wav_np = self._audio_tokenizer.detokenize(
            pred_global_ids.to(self._device).squeeze(0),  # Shape: (1, N_global)
            pred_semantic_ids.to(self._device),  # Shape: (1, N_semantic)
        )

        return wav_np

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[SingleSpeaker] = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_audio_tokens: int = 2048,
        output_filename: str = None,
    ) -> InferenceOutput:
        """
        Generate speech from text with optional speaker ID.

        Args:
            text (str): Input text in Bambara to convert to speech.
            speaker_id (str, optional): Speaker identifier (e.g., "SPEAKER_01", "SPEAKER_18").
            temperature (float): Sampling temperature (default: 0.8).
            top_k (int): Top-k sampling parameter (default: 50).
            top_p (float): Top-p sampling parameter (default: 1.0).
            max_new_audio_tokens (int): Maximum audio tokens to generate (default: 2048).
            output_filename (str, optional): Name of output audio file.

        Returns:
            np.ndarray: Generated waveform as a NumPy array.
        """

        # I'm avoiding this defensive check to allow to use this inference class for other spark_base tts whiout changing the any code
        # if speaker_id.id.upper() not in Settings.speakers_ids :
        #     raise ValueError("This speaker is not supported")

        if not text:
            raise ValueError("text can not be empty")

        if not isinstance(text, str):
            raise TypeError("text should be a string")

        formatted_text = f"{speaker_id.id}: {text}" if speaker_id else text

        try:
            generated_waveform = self._generate_speech_from_text(
                text=formatted_text,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_audio_tokens=max_new_audio_tokens,
            )
            sample_rate = self._audio_tokenizer.config.get("sample_rate", 16000)

            if generated_waveform.size > 0 and output_filename:
                sf.write(output_filename, generated_waveform, sample_rate)

            return InferenceOutput(
                waveform=generated_waveform,
                sample_rate=sample_rate,
                output_filename=output_filename,
            )

        except Exception as e:
            return InferenceOutput(error_message=f"error when generating speech: {e}")


# if __name__ == "__main__":
#     try:
#         from maliba_ai.settings.tts.bam_spark import Speakers
#         tts = BamSparkTTS()


#         examples = {
#             Speakers.Adama: "An filɛ ni ye yɔrɔ minna ni an ye an sigi ka a layɛ yala an bɛ ka baara min kɛ ɛsike a kɛlen don ka Ɲɛ wa ?",
#             Speakers.Moussa: "An filɛ ni ye yɔrɔ minna ni an ye an sigi ka a layɛ yala an bɛ ka baara min kɛ ɛsike a kɛlen don ka Ɲɛ wa ?"
#         }

#         for speaker_id, text in examples.items():
#             speaker_id = SingleSpeaker(
#                 id=speaker_id
#             )

#             output = tts.synthesize(
#                 text=text,
#                 speaker_id=speaker_id,
#                 output_filename=f"test_{speaker_id.id}.wav"
#             )

#             if output.error_message:
#                 print({output.error_message})


#     except Exception as e:
#         print(e)
