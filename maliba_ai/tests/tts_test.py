# library used by test
import numpy as np
import pytest

# class to test
# SparkTTS
from maliba_ai.core.tts.models.sparktts.inference import BamSparkTTS
# vitsTTS
# To generate data samples
from maliba_ai.tests.inference_data_samples import inference_data_samples


def test_sparktts(data_samples):
    spark_tts = BamSparkTTS()

    # test 1 - no text provided
    # result: A ValueError is raised
    with pytest.raises(ValueError):
        output = spark_tts.synthesize(text="", output_filename="no_text_test.wav")

    # test 2 - non string provided
    # result: A TypeError is raised
    with pytest.raises(TypeError):
        output = spark_tts.synthesize(
            text=0, output_filename="non_string_text_test.wav"
        )

    # test 3 - successful audio generations
    # result: An InferenceOutput object should be returned whith no error message
    for data_sample in enumerate(data_samples):
        output = spark_tts.synthesize(
            text=data_sample[1][0],
            speaker_id=data_sample[1][1],
            output_filename=f"spark_tts_output_{data_sample[0]}.wav",
        )
        assert output.error_message is None
        assert output.output_filename == f"spark_tts_output_{data_sample[0]}.wav"
        assert isinstance(output.waveform, np.ndarray)


def test_vits_tts():
    pass


if __name__ == "__main__":
    data_samples = inference_data_samples("TTS")
    test_sparktts(data_samples)
