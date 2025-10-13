import os
import tempfile
from typing import List, Optional

import numpy as np
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity

from maliba_ai.settings.embeddings.fasttext import \
    Settings as embeddings_settings
from maliba_ai.settings.embeddings.settings import (EmbeddingOutput,
                                                    SimilarityOutput,
                                                    SimilarWordsOutput,
                                                    TextSearchOutput)


class WordsEmbeddings:
    """
    FastText word embeddings wrapper for the Bambara language.

    Provides:
    - Loading pre-trained embeddings from Hugging Face or local files
    - Word vector retrieval
    - Similarity computations
    - Text-to-vector conversion
    - Text similarity search
    """

    def __init__(
        self,
        model_id: str = embeddings_settings.model_repo,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.model: Optional[KeyedVectors] = None
        self.vocab: Optional[List[str]] = None
        self.dimension: Optional[int] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            model_path = hf_hub_download(
                repo_id=self.model_id,
                filename=embeddings_settings.model_file_name,
                cache_dir=self.cache_dir,
            )
            self.model = KeyedVectors.load(model_path)
            self._initialize_model_attributes()
        except Exception:
            self._load_from_local()

    def _load_from_local(self) -> None:
        local_paths = embeddings_settings.local_paths
        for path in local_paths:
            if os.path.exists(path):
                try:
                    self.model = KeyedVectors.load(path)
                    self._initialize_model_attributes()
                    return
                except Exception:
                    continue
        raise RuntimeError("Failed to load model from Hugging Face or local files.")

    def _initialize_model_attributes(self) -> None:
        if self.model is None:
            raise ValueError("Model is not loaded.")
        self.vocab = list(self.model.key_to_index.keys())
        self.dimension = self.model.vector_size

    def get_word_vector(self, word: str) -> EmbeddingOutput:
        if not self.contains_word(word):
            return EmbeddingOutput(error_message=f"Word '{word}' not in vocabulary.")
        return EmbeddingOutput(vector=self.model[word])

    def get_similar_words(self, word: str, top_k: int = 10) -> SimilarWordsOutput:
        if not self.contains_word(word):
            return SimilarWordsOutput(error_message=f"Word '{word}' not in vocabulary.")
        return SimilarWordsOutput(words=self.model.most_similar(word, topn=top_k))

    def calculate_similarity(self, word1: str, word2: str) -> SimilarityOutput:
        if not (self.contains_word(word1) and self.contains_word(word2)):
            return SimilarityOutput(
                error_message=f"One of '{word1}' or '{word2}' not in vocabulary."
            )
        vec1 = self.model[word1]
        vec2 = self.model[word2]
        score = float(cosine_similarity([vec1], [vec2])[0][0])
        return SimilarityOutput(score=score)

    def text_to_vector(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vectors = [self.model[word] for word in words if self.contains_word(word)]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.dimension)

    def search_similar_texts(
        self, query: str, texts: List[str], top_k: int = 5
    ) -> TextSearchOutput:
        try:
            query_vector = self.text_to_vector(query)
            similarities = []
            for idx, text in enumerate(texts):
                text_vector = self.text_to_vector(text)
                if np.any(text_vector):
                    score = float(
                        cosine_similarity([query_vector], [text_vector])[0][0]
                    )
                    similarities.append((score, text, idx))
            similarities.sort(key=lambda x: x[0], reverse=True)
            return TextSearchOutput(results=similarities[:top_k])
        except Exception as e:
            return TextSearchOutput(error_message=str(e))

    def get_vocabulary(self) -> List[str]:
        return self.vocab.copy() if self.vocab else []

    def contains_word(self, word: str) -> bool:
        return self.model is not None and word in self.model.key_to_index


# if __name__ == "__main__":
#     embeddings = WordsEmbeddings()

#     sim_words = embeddings.get_similar_words("ye", top_k=10)
#     if sim_words.error_message:
#         print("Error:", sim_words.error_message)
#     else:
#         print("Similar words:", sim_words.words)
