from typing import List, Optional

from maliba_ai.settings.llm.settings import GenerationOutput, Settings

# Supported LLM architectures (matching maliba_ai.settings.settings.Models).
GEMMA = "gemma-3n"


class LLM:
    """
    Unified entry point for text generation with Bambara language models.

    Dispatches to the underlying model wrapper based on the chosen
    architecture while exposing `generate` and `chat` interfaces.

    Example:
        llm = LLM()
        llm.generate("I ni ce, ...")
        llm.chat([{"role": "user", "content": "I ni ce"}])
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        architecture: str = GEMMA,
        max_new_tokens: int = Settings.max_new_tokens,
    ) -> None:
        self.architecture = architecture

        if architecture in (GEMMA, "gemma"):
            from maliba_ai.core.llm.models.gemma_base import GemmaLLM

            self._model = GemmaLLM(
                model_id or Settings.model_repo, max_new_tokens=max_new_tokens
            )
        else:
            raise ValueError(
                f"Unsupported LLM architecture: {architecture}. Available: {GEMMA}"
            )

    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        """Generate a completion for a single free-form prompt."""
        return self._model.generate(prompt, **kwargs)

    def chat(self, messages: List[dict], **kwargs) -> GenerationOutput:
        """Generate a response for a chat conversation."""
        return self._model.chat(messages, **kwargs)
