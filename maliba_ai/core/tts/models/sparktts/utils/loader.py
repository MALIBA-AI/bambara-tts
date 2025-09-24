
import os
import torch
from unsloth import FastModel
from huggingface_hub import snapshot_download
from maliba_ai.settings.tts.spark import Settings
from maliba_ai.core.tts.models.sparktts.models.audio_tokenizer import BiCodecTokenizer

def load_tts_model(model_path:str = Settings.model_repo, max_seq_length:int = 2048):
    """
    Load the TTS model and tokenizer from the specified repository.
    
    Args:
        model_path: Model path (local or on Hugging Face).
        max_seq_length : the max seq lenght 
    
    Returns:
        tuple: (model, tokenizer) - Loaded TTS model and tokenizer.
    """
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=torch.float32,
        load_in_4bit=False
    )
    FastModel.for_inference(model)
    return model, tokenizer


def load_audio_tokenizer(device):
    """
    Load the audio tokenizer, downloading the base model if necessary.
    
    Args:
        device (torch.device): Device to load the tokenizer on ('cuda' or 'cpu').
    
    Returns:
        BiCodecTokenizer: Loaded audio tokenizer instance.
    """
    if not os.path.exists("Spark-TTS-0.5B"):
        snapshot_download(Settings.base_spark_model, local_dir="Spark-TTS-0.5B")
    audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", device)
    return audio_tokenizer