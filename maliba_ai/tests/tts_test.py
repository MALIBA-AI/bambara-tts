# library used by test
import pytest 
import numpy as np

# To generate data samples
import inference_data_samples

# class to test
# SparkTTS
from maliba_ai.core.tts.models.sparktts.inference import BamSparkTTS
# vitsTTS
from maliba_ai.core.tts.models.vits.inference import MalianTTS

def test_sparktts():
    spark_tts = BamSparkTTS()

    # test 1 - no text provided
    # result: A ValueError is raised
    with pytest.raises(ValueError):
        output = spark_tts.synthesize(
            text="",
            output_filename = f"no_text_test.wav"
        ) 
        

    # test 2 - non string provided
    # result: A TypeError is raised
    with pytest.raises(TypeError):
        output = spark_tts.synthesize(
            text=0,
            output_filename = f"non_string_text_test.wav"
        ) 

    # test 3 - successful audio generations
    # result: An InferenceOutput object should be returned whith no error message
    for data_sample in enumerate(inference_data_samples('TTS')):
        output = spark_tts.synthesize(
            text=data_sample[1][0],
            speaker_id=data_sample[1][1],
            output_filename = f"spark_tts_output_{data_sample[0]}.wav"
        ) 
        assert output.error_message is None
        assert output.output_filename == f"spark_tts_output_{data_sample[0]}.wav"
        assert isinstance(output.waveform, np.ndarray)

def test_vits_tts():
    vits_tts = MalianTTS()

    




if __name__ == "__main__":
    test_sparktts()