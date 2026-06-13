from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from maliba_ai.settings.llm.settings import GenerationOutput, Settings


class GemmaLLM:
    """
    Text generation wrapper around a Gemma-based causal language model.

    Provides:
    - Loading an instruction-tuned Gemma model from Hugging Face
    - Single-prompt completion (`generate`)
    - Multi-turn chat using the tokenizer chat template (`chat`)
    """

    def __init__(
        self,
        model_id: str = Settings.model_repo,
        max_new_tokens: int = Settings.max_new_tokens,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=self.torch_dtype,
        )

    def _generate(self, inputs, **kwargs) -> str:
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": Settings.temperature,
            "top_k": Settings.top_k,
            "top_p": Settings.top_p,
        }
        generation_kwargs.update(kwargs)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]

    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        """
        Generate a completion for a single free-form prompt.

        Args:
            prompt (str): Input prompt.
            **kwargs: Optional generation overrides (temperature, top_k, ...).

        Returns:
            GenerationOutput: Generated text or an error message.
        """
        if not prompt:
            return GenerationOutput(error_message="prompt cannot be empty")

        if not isinstance(prompt, str):
            return GenerationOutput(error_message="prompt should be a string")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            return GenerationOutput(text=self._generate(inputs, **kwargs))
        except Exception as e:
            return GenerationOutput(error_message=f"error during generation: {e}")

    def chat(self, messages: List[dict], **kwargs) -> GenerationOutput:
        """
        Generate a response for a chat conversation.

        Args:
            messages (List[dict]): List of {"role": ..., "content": ...} messages.
            **kwargs: Optional generation overrides.

        Returns:
            GenerationOutput: Assistant reply or an error message.
        """
        if not messages:
            return GenerationOutput(error_message="messages cannot be empty")

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            return GenerationOutput(text=self._generate(inputs, **kwargs))
        except Exception as e:
            return GenerationOutput(error_message=f"error during chat: {e}")
