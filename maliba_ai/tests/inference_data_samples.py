from typing import Optional, Union

from maliba_ai.settings.tts.bam_spark import Speakers


def inference_data_samples(
    test_type: str,
    text_samples: Optional[list[str]] = [],
    speakers: Optional[list[Speakers]] = [],
    audio_samples: Optional[list[Union[str, bytes]]] = [],
    audio_paths: Optional[list[str]] = [],
) -> Union[list[tuple[str, Speakers]], list[bytes], list[str]]:
    """
    Generate data samples for inference tests

    Args:
        test_type (str): Type of model to be tested: TTS, ASR, MT or LLM
        text_samples (optional): text samples to use for the test
        speakers (for TTS test): speakers to use for the test
        audio_samples (for ASR): audio samples to use for the test
    Returns:
        data samples with the correct data types depending on the type
        of model being tested
    """

    # Data samples to return
    data_samples = []

    # Building data_samples based on the test type
    if test_type == "TTS":
        # input: (Text, Speaker)
        # speakers to include in the tests if none specified
        if speakers is None or len(speakers) == 0:
            speakers = [
                Speakers.Adama,
                Speakers.Moussa,
                Speakers.Bourama,
                Speakers.Modibo,
                Speakers.Seydou,
                Speakers.Amadou,
                Speakers.Bakary,
                Speakers.Ngolo,
                Speakers.Amara,
                Speakers.Ibrahima,
            ]
            # text samples to include in the tests if none specified
        if text_samples is None or len(text_samples) == 0:
            text_samples = [
                "An filɛ ni ye yɔrɔ minna ni an ye an sigi ka a layɛ yala an bɛ ka baara min kɛ ɛsike a kɛlen don ka Ɲɛ wa ?"
            ]
        for text in text_samples:
            for speaker in speakers:
                data_samples.append((text, speaker))
    elif test_type == "ASR":
        # input: Audio file path
        data_samples = ["./audio_samples/"]
    elif test_type == "MT":
        # input: Text
        # text samples to include in the tests if none specified
        if text_samples is None or len(text_samples) == 0:
            # TODO: Add more samples
            text_samples = [
                "An filɛ ni ye yɔrɔ minna ni an ye an sigi ka a layɛ yala an bɛ ka baara min kɛ ɛsike a kɛlen don ka Ɲɛ wa ?"
            ]
        data_samples = text_samples
    elif test_type == "LLM":
        # input: Text
        # text samples to include in the tests if none specified
        if text_samples is None or len(text_samples) == 0:
            # TODO: Add more samples
            text_samples = [
                "An filɛ ni ye yɔrɔ minna ni an ye an sigi ka a layɛ yala an bɛ ka baara min kɛ ɛsike a kɛlen don ka Ɲɛ wa ?"
            ]
        data_samples = text_samples
    else:
        raise ValueError("Test type can only be one of these: TTS, ASR, MT, LLM.")
    return data_samples


def test(type):
    for data in inference_data_samples(type):
        print(data)


if __name__ == "__main__":
    test("LLM")
