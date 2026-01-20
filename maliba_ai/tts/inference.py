import torch
import numpy as np
import re
import soundfile as sf

from maliba_ai.models.models import load_tts_model, load_audio_tokenizer
from maliba_ai.config.speakers import Adama, SingleSpeaker, Settings
from typing import  Optional

from bambara_normalizer import normalize

class BambaraTTSInference:
    def __init__(self, model_path:Optional[str] = Settings.model_repo, max_seq_length:Optional[int] = 2048):
        """
        Initialize the Bambara TTS inference class.
        
        Args:
            model_path (str, optional): Path to the model (local or huggingface repo id).
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._tokenizer = load_tts_model(model_path=model_path, max_seq_length=max_seq_length)
        self._audio_tokenizer = load_audio_tokenizer(self._device)
    

    @torch.inference_mode()
    def _generate_speech_from_text(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_audio_tokens: int = 2048,
    ) -> np.ndarray:
        """
        Generate speech from pre-formatted text.
        
        Args:
            text (str): Pre-formatted text (with speaker ID if applicable).
            temperature (float): Sampling temperature (default: 0.8).
            top_k (int): Top-k sampling parameter (default: 50).
            top_p (float): Top-p sampling parameter (default: 1.0).
            max_new_audio_tokens (int): Maximum audio tokens to generate (default: 2048).
            
        Returns:
            np.ndarray: Generated waveform as a NumPy array.
        """
        prompt = "".join([
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>"
        ])
        
        model_inputs = self._tokenizer([prompt], return_tensors="pt").to(self._device)
    
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=max_new_audio_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id
        )

        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
        predicted_text = self._tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
    
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_text)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicted_text)
        
        if not semantic_matches:
            return np.array([], dtype=np.float32)
        
        pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
        
        if not global_matches:
            pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
        else:
            pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0)
        
        pred_global_ids = pred_global_ids.unsqueeze(0)  # Shape: (1, 1, N_global)
        
        self._audio_tokenizer.device = self._device
        self._audio_tokenizer.model.to(self._device)
        
        wav_np = self._audio_tokenizer.detokenize(
            pred_global_ids.to(self._device).squeeze(0),  # Shape: (1, N_global)
            pred_semantic_ids.to(self._device)            # Shape: (1, N_semantic)
        )
        
        return wav_np


    def generate_speech(
        self,
        text: str,
        speaker_id:Optional[SingleSpeaker]  = Adama,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        max_new_audio_tokens: int = 2048,
        output_filename: str = None
    ) -> np.ndarray:
        
        """
        Generate speech from text with optional speaker ID.
        
        Args:
            text (str): Input text in Bambara to convert to speech.
            speaker_id (str, optional): Speaker identifier (e.g., "SPEAKER_01", "SPEAKER_18").
            temperature (float): Sampling temperature (default: 0.8).
            top_k (int): Top-k sampling parameter (default: 50).
            top_p (float): Top-p sampling parameter (default: 1.0).
            max_new_audio_tokens (int): Maximum audio tokens to generate (default: 2048).
            output_filename (str, optional): Name of output audio file.
            
        Returns:
            np.ndarray: Generated waveform as a NumPy array.
        """

        #let's normalized the input text here we use : 
        # https://pypi.org/project/bambara-text-normalizer
        
        text = normalize(
            text,
            mode="expand",  
            preserve_tones=False,               
            normalize_legacy_orthography=True, 
            lowercase=True,                     
            remove_punctuation=False,           
            normalize_whitespace=True,         
            normalize_apostrophes=True,         
            normalize_special_chars=True,    
            expand_dates = True,
            expand_numbers=True,  
            expand_times=True,            
            remove_diacritics_except_tones=False,  
            handle_french_loanwords=True,   
            strip_repetitions=False,       
            normalize_compounds=True, 
        )


        if speaker_id.id.upper() not in Settings.speakers_ids : 
            raise ValueError("This speaker is not supported")
        
        if not text : 
            raise ValueError("text can not be empty")
        
        if not isinstance(text, str):
            raise TypeError("text should be a string")
                
        formatted_text = f"{speaker_id.id}: " + text  if speaker_id else text
        generated_waveform = self._generate_speech_from_text(
            text=formatted_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_audio_tokens=max_new_audio_tokens
        )
        
        if generated_waveform.size > 0 and output_filename:
            sample_rate = self._audio_tokenizer.config.get("sample_rate", 16000)
            sf.write(output_filename, generated_waveform, sample_rate)
        
        return generated_waveform