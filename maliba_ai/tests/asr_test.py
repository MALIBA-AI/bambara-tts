# library used by test
import pytest

# class to test
# whisper base ASR
from maliba_ai.core.asr.models.whisper_base import ASR
# To generate data samples
from maliba_ai.tests.inference_data_samples import inference_data_samples


# whisper base ASR
def test_whisper_base_asr(model_id: str):
    whisper_base_asr = ASR(model_id)

    # test 1 - successful audio transcription with default parameters
    # result: an output dict containing the transcription should be returned
    for data_sample in inference_data_samples("ASR"):
        output = whisper_base_asr.transcribe(data_sample)
        assert output is not None
        # Printing the output dict content
        print("ASR Output")
        for key, value in zip(output.keys(), output.values()):
            print(key, value)
            print(" ===================== =====================")
        # TODO : Check output dict content
        # output shape {'text' : 'A ye kuma'}
        assert "text" in output.keys()

        # test 2 - successful audio transcription with personalized parameters
        # result: an output dict containing the transcription should be returned
        for data_sample in inference_data_samples("ASR"):
            output = whisper_base_asr.transcribe(
                data_sample,
                temperature=0.25,
                repetition_penalty=1.1,
            )
            assert isinstance(output, dict)
            # Printing the output dict content
            print("ASR Output")
            for key, value in zip(output.keys(), output.values()):
                print(key, value)
                print(" ===================== =====================")
            # TODO : Check output dict content

    # test 3 - no audio provided
    # result: an error is raised
    with pytest.raises(TypeError):
        error_output = whisper_base_asr.transcribe()
    assert error_output is None

    # test 4 - wrong filepath provided
    # result: an error is raised
    with pytest.raises(FileNotFoundError):
        error_output = whisper_base_asr.transcribe("no_file.wav")
    assert error_output is None


def test_gemma_base_asr():
    pass


if __name__ == "__main__":
    test_whisper_base_asr("MALIBA-AI/bambara-asr-v2")
