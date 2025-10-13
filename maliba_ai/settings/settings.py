from dataclasses import dataclass
from typing import List


@dataclass
class Model:
    architecture: str
    task: str
    version: str
    path: str


class AvailableASRModels:
    """
    Class to hold available ASR models and their configurations.
    """

    bambara: List[Model] = [
        Model(
            architecture="whisper",
            task="asr",
            version="v1",
            path="MALIBA-AI/bambara-asr-v{version}",
        ),
        Model(
            architecture="gemma-3n",
            task="asr",
            version="v1",
            path="MALIBA-AI/gemma-asr-v{version}",
        ),
    ]

    songhoy: List[Model] = [
        Model(
            architecture="whisper",
            task="asr",
            version="v1",
            path="MALIBA-AI/songhoy-asr-v{version}",
        )
    ]

    bomu: List[Model] = [
        Model(
            architecture="wav2vec",
            task="asr",
            version="v1",
            path="MALIBA-AI/bomu-asr-v{version}",
        )
    ]
    dogon: List[Model] = [
        Model(
            architecture="wav2vec",
            task="asr",
            version="v1",
            path="MALIBA-AI/dogon-asr-v{version}",
        )
    ]
    minianka: List[Model] = [
        Model(
            architecture="wav2vec",
            task="asr",
            version="v1",
            path="MALIBA-AI/minianka-asr-v{version}",
        )
    ]


class AvailableTTSModels:
    bam_spark: List[Model] = [
        Model(
            architecture="spark_tts",
            task="tts",
            version="v1",
            path="MALIBA-AI/bambara-tts-v{version}",
        )
    ]
    malian_tts: List[Model] = [
        Model(
            architecture="vits",
            task="tts",
            version="v1",
            path="MALIBA-AI/malian-tts-v{version}",
        )
    ]
    multilangual_tts: List[Model] = [
        Model(
            architecture="spark_tts",
            task="tts",
            version="v1",
            path="MALIBA-AI/multilingual-tts-v{version}",
        )
    ]


class AvailableEmbeddingModels:
    bambara: List[Model] = [
        Model(
            architecture="fasttext",
            task="embeddings",
            version="v1",
            path="MALIBA-AI/bambara-embeddings-v{version}",
        )
    ]


class AvailableLLMModels:
    bambara: List[Model] = [
        Model(
            architecture="gemma-3n",
            task="llm",
            version="v1",
            path="MALIBA-AI/bambara-llm-v{version}",
        )
    ]


class AvailableMTModels:
    bambara: List[Model] = [
        Model(
            architecture="nllb",
            task="mt",
            version="v1",
            path="MALIBA-AI/bambara-mt-{version}",
        )
    ]


class Models:
    asr = AvailableASRModels
    tts = AvailableTTSModels
    mt = AvailableMTModels
    llm = AvailableLLMModels
    embeddings = AvailableEmbeddingModels

    @classmethod
    def get(cls, category: str):
        return getattr(cls, category, None)


@dataclass
class Settings:
    models: Models = Models
