from dataclasses import dataclass
from typing import Dict, List


class Settings:
    model_repo: str = "MALIBA-AI/bambara-tts"
    gguf_repo: str = "MALIBA-AI/bambara-tts-gguf"
    base_spark_model: str = "unsloth/Spark-TTS-0.5B"
    sample_rate: int = 16000
    speakers_ids: List[str] = [
        "SPEAKER_1", "SPEAKER_2", "SPEAKER_3", "SPEAKER_4", "SPEAKER_5",
        "SPEAKER_6", "SPEAKER_7", "SPEAKER_8", "SPEAKER_9", "SPEAKER_10",
    ]


# Single source of truth for Bambara text normalization, shared by the local
# PyTorch inference path and the llama.cpp inference server so both produce the
# exact same model input. See https://pypi.org/project/bambara-text-normalizer
NORMALIZE_OPTIONS: Dict[str, object] = {
    "mode": "expand",
    "preserve_tones": False,
    "normalize_legacy_orthography": True,
    "lowercase": True,
    "remove_punctuation": False,
    "normalize_whitespace": True,
    "normalize_apostrophes": True,
    "normalize_special_chars": True,
    "expand_dates": True,
    "expand_numbers": True,
    "expand_times": True,
    "remove_diacritics_except_tones": False,
    "handle_french_loanwords": True,
    "strip_repetitions": False,
    "normalize_compounds": True,
}


@dataclass
class SingleSpeaker:
    id: str

    def __post_init__(self):
        if self.id not in Settings.speakers_ids:
            raise ValueError(
                f"Invalid speaker '{self.id}'. Valid speakers: {Settings.speakers_ids}"
            )

    def __str__(self) -> str:
        return f"Speaker({self.id})"

    def __repr__(self) -> str:
        return f"SingleSpeaker(id='{self.id}')"


class Speakers:
    Adama: SingleSpeaker     = SingleSpeaker(id="SPEAKER_1")
    Moussa: SingleSpeaker    = SingleSpeaker(id="SPEAKER_2")
    Bourama: SingleSpeaker   = SingleSpeaker(id="SPEAKER_3")
    Modibo: SingleSpeaker    = SingleSpeaker(id="SPEAKER_4")
    Seydou: SingleSpeaker    = SingleSpeaker(id="SPEAKER_5")
    Amadou: SingleSpeaker    = SingleSpeaker(id="SPEAKER_6")
    Bakary: SingleSpeaker    = SingleSpeaker(id="SPEAKER_7")
    Ngolo: SingleSpeaker     = SingleSpeaker(id="SPEAKER_8")
    Amara: SingleSpeaker     = SingleSpeaker(id="SPEAKER_9")
    Ibrahima: SingleSpeaker  = SingleSpeaker(id="SPEAKER_10")

    @classmethod
    def get_all_speakers(cls) -> List[SingleSpeaker]:
        return [
            cls.Adama, cls.Moussa, cls.Bourama, cls.Modibo, cls.Seydou,
            cls.Amadou, cls.Bakary, cls.Ngolo, cls.Amara, cls.Ibrahima,
        ]

    @classmethod
    def get_speaker_by_name(cls, name: str) -> SingleSpeaker:
        speaker = getattr(cls, name, None)
        if isinstance(speaker, SingleSpeaker):
            return speaker
        available = [s.id for s in cls.get_all_speakers()]
        raise ValueError(f"Speaker '{name}' not found. Available: {available}")

    @classmethod
    def get_speaker_by_id(cls, speaker_id: str) -> SingleSpeaker:
        """Resolve a raw id like 'SPEAKER_3' (case-insensitive) to a SingleSpeaker."""
        target = speaker_id.upper()
        for speaker in cls.get_all_speakers():
            if speaker.id.upper() == target:
                return speaker
        raise ValueError(
            f"Speaker id '{speaker_id}' not found. Available: {Settings.speakers_ids}"
        )
