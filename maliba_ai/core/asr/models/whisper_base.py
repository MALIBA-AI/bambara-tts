import torch
from  typing import Union, Optional

from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    pipeline
)


class ASR:
    """
    Automatic Speech Recognition (ASR) wrapper using OpenAI's Whisper models base.

    This class provides an easy-to-use interface for maliba-ai ASR based on Whisper models.:
    - Loading  models from Hugging Face
    - Configuring the ASR pipeline
    - Transcribing audio into text
    """

    def __init__(self, model_id: str )->None:
        self.model_id = model_id
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.pipe = None
        self.load_model()
        self.configure_pipeline()

    def load_model(self):
        """
        Load the Whisper model, tokenizer, and processor from Hugging Face Hub.

        Configures model settings such as:
        - Beam search size
        - Maximum sequence length
        - Repetition penalty
        """
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            use_cache=True,
            attention_dropout=0.1,
            dropout=0.1
        )
        
        self.model.config.suppress_tokens = []
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.early_stopping = True
        self.model.config.max_length = 448
        self.model.config.num_beams = 5

        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_id)
        self.processor = WhisperProcessor.from_pretrained(self.model_id)

    def configure_pipeline(self):
        """
        Configure the Hugging Face ASR pipeline with the loaded model.

        Uses Whisper with:
        - 30s audio chunks
        - 3s stride
        - Configurable dtype (float16 if CUDA available)
        """
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            chunk_length_s=30,
            stride_length_s=3,
            return_timestamps=False,
            batch_size=1
        )

    def transcribe(self, 
                   audio: Union[str, bytes], 
                   temperature: Optional[float] = 0.0, 
                   do_sample: Optional[bool] = False, 
                   length: Optional[int] = 1.0, 
                   repetition_penalty: Optional[float] = 1.2, 
                     num_beams: Optional[int] = 5
                   ) -> dict:
        
        """
        Transcribe audio into text using the Whisper ASR pipeline.

        Args:
            audio (str or bytes): Path to audio file or raw audio bytes.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            do_sample (bool, optional): Enable sampling instead of greedy decoding. Defaults to False.
            length (float, optional): Length penalty (discourages too short/long outputs). Defaults to 1.0.
            repetition_penalty (float, optional): Penalizes repeated sequences. Defaults to 1.2.
            num_beams (int, optional): Beam search width. Defaults to 5.

        Returns:
            dict: Transcription result with recognized text.
            None: If transcription fails.
        """
        try:
            result = self.pipe(
                audio,
                generate_kwargs={
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "num_beams": num_beams,
                    "length_penalty": length,
                    "repetition_penalty": repetition_penalty
                }
            )
            return result
        
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None