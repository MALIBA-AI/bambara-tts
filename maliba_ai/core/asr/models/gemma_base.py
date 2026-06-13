from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from maliba_ai.settings.asr.settings import Settings


class GemmaASR:
    """
    Automatic Speech Recognition wrapper using an audio-capable Gemma-3n model.

    Unlike the Whisper wrapper, Gemma-3n is an instruction-tuned multimodal model:
    transcription is performed by prompting the model with the audio plus a
    transcription instruction. The public interface mirrors the Whisper `ASR`
    wrapper (`transcribe(audio) -> {"text": ...}`) so both can be used
    interchangeably through the ASR inference dispatcher.
    """

    def __init__(
        self,
        model_id: str = Settings.gemma_model_repo,
        prompt: str = Settings.transcription_prompt,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.torch_dtype = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=self.torch_dtype,
        )

    def transcribe(
        self,
        audio: Union[str, bytes],
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> dict:
        """
        Transcribe audio into text.

        Args:
            audio (str or bytes): Path to an audio file or raw audio bytes.
            prompt (str, optional): Override the default transcription instruction.
            max_new_tokens (int, optional): Maximum tokens to generate.

        Returns:
            dict: Transcription result of shape {"text": ...}.
            None: If transcription fails.
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": prompt or self.prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, **kwargs
                )

            trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
            text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            return {"text": text.strip()}

        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
