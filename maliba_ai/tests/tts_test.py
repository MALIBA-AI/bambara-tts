# library used by test
import pytest 
import numpy as np

import inference_data_samples
# class to test
# SparkTTS
from maliba_ai.core.tts.models.sparktts.inference import BamSparkTTS

# SparkTTS
spark_tts = BamSparkTTS() 

def test_sparktts():
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
    for data_sample in enumerate(inference_data_samples):
        output = spark_tts.synthesize(
            text=data_sample[1][0],
            speaker_id=data_sample[1][1],
            output_filename = f"spark_tts_output_{data_sample[O]}.wav"
        ) 
        assert output.error_message == None
        assert output.output_filename == f"spark_tts_output_{data_sample[O]}.wav"
        assert type(output.waveform) is np.ndarray

def test_vits_tts():
    pass