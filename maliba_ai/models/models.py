
import os
import torch
from huggingface_hub import snapshot_download
from maliba_ai.config.settings import Settings
from maliba_ai.sparktts.models.audio_tokenizer import BiCodecTokenizer


def load_tts_model(
    model_path: str = Settings.model_repo,
    max_seq_length: int = 2048,
    dtype: torch.dtype = torch.float32,
):
    """
    Load the TTS model and tokenizer from the specified repository.

    Args:
        model_path: Model path (local or on Hugging Face).
        max_seq_length: The max sequence length.
        dtype: Compute dtype for the model weights (default float32 for quality).

    Returns:
        tuple: (model, tokenizer) - Loaded TTS model and tokenizer.
    """
    # Imported lazily: unsloth is a heavy, optional dependency and is only needed
    # for the local PyTorch generation path (not for the llama.cpp server).
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
    )
    FastModel.for_inference(model)
    return model, tokenizer


def download_spark_assets(model_dir: str = "Spark-TTS-0.5B", detokenize_only: bool = False) -> str:
    """Download the Spark-TTS assets needed for the BiCodec tokenizer.

    Args:
        model_dir: Local directory to download into.
        detokenize_only: If True, only fetch what ``detokenize`` needs (config +
            BiCodec checkpoint) and skip the large wav2vec2 feature extractor.

    Returns:
        The local model directory path.
    """
    if detokenize_only:
        # Only the root config.yaml and the BiCodec checkpoint are required to
        # turn tokens back into audio.
        allow_patterns = ["config.yaml", "BiCodec/*"]
        ready = os.path.exists(os.path.join(model_dir, "BiCodec", "model.safetensors"))
    else:
        allow_patterns = None
        ready = os.path.exists(os.path.join(model_dir, "wav2vec2-large-xlsr-53"))

    if not ready:
        snapshot_download(
            Settings.base_spark_model,
            local_dir=model_dir,
            allow_patterns=allow_patterns,
        )
    return model_dir


def load_audio_tokenizer(device, detokenize_only: bool = False, model_dir: str = "Spark-TTS-0.5B"):
    """
    Load the audio tokenizer, downloading the base model if necessary.

    Args:
        device (torch.device): Device to load the tokenizer on ('cuda' or 'cpu').
        detokenize_only: If True, load only the decoding path (no wav2vec2). Use
            this for a synthesis-only server to save memory and download time.
        model_dir: Local directory holding the Spark-TTS assets.

    Returns:
        BiCodecTokenizer: Loaded audio tokenizer instance.
    """
    download_spark_assets(model_dir=model_dir, detokenize_only=detokenize_only)
    return BiCodecTokenizer(model_dir, device, load_wav2vec2=not detokenize_only)
