from typing import Optional

# Supported embedding architectures (matching maliba_ai.settings.settings.Models).
FASTTEXT = "fasttext"


class Embeddings:
    """
    Unified entry point for word embeddings.

    Dispatches to the underlying model wrapper based on the chosen
    architecture and transparently delegates the embedding methods
    (`get_word_vector`, `get_similar_words`, `calculate_similarity`,
    `text_to_vector`, `search_similar_texts`, ...).

    Example:
        emb = Embeddings()
        emb.get_similar_words("ye", top_k=10)
    """

    def __init__(
        self,
        architecture: str = FASTTEXT,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.architecture = architecture

        if architecture == FASTTEXT:
            from maliba_ai.core.embeddings.models.fasttext_base import WordsEmbeddings
            from maliba_ai.settings.embeddings.fasttext import Settings

            self._model = WordsEmbeddings(
                model_id or Settings.model_repo, cache_dir=cache_dir
            )
        else:
            raise ValueError(
                f"Unsupported embeddings architecture: {architecture}. "
                f"Available: {FASTTEXT}"
            )

    def __getattr__(self, name):
        # Delegate any embedding method to the underlying model wrapper.
        return getattr(self.__dict__["_model"], name)
