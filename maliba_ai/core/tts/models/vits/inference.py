import torch
from transformers import VitsModel, AutoTokenizer
from typing import  Optional, Dict
from maliba_ai.settings.tts.malian_tts import Settings
import soundfile as sf

from maliba_ai.settings.tts.settings import InferenceOutput

class MalianTTS:
    """
    Wrapper for multilingual Malian TTS (a collection of VITS models: (MMS fine-tuned )) (Bambara, Boomu, Dogon, Pular, Songhoy, Tamasheq).
    Uses Hugging Face VITS models fine-tuned for six Malian languages.
    """

    def __init__(self, model_id: str = Settings.models_repo, languages: list[str] = Settings.models) -> None:
        self.model_id = model_id
        self.models: Dict[str, VitsModel] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.device = self._get_device()
        self.languages = languages 
        self._load_models()

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_models(self) -> None:
        if any(lang not in Settings.models for lang in self.languages):
            raise ValueError(f"Unsupported language in {self.languages}. Available: {Settings.models}")
        
         
        for lang in self.languages:
            self.models[lang] = VitsModel.from_pretrained(
                self.model_id,
                subfolder=f"{Settings.models_subfolder}/{lang}"
            ).to(self.device)

            self.tokenizers[lang] = AutoTokenizer.from_pretrained(
                self.model_id,
                subfolder=f"{Settings.models_subfolder}/{lang}"
            )

    def synthesize(
        self, 
        text: str,
        language: Optional[str] = None,
        output_filename: str = None
    ) -> InferenceOutput:
        """
        Generate audio from text for the given language.

        Returns:
            (sample_rate, waveform), None if successful
            None, error_message if failed
        """
        language = language or self.languages[0] # why the first one: if the user load only one model so no need to specify the language

        if language not in self.models:
            raise ValueError(f"Unsupported language: {language}. Available: {self.languages}")

        if not text.strip():
            return None, "Please enter some text to synthesize."

        try:
            model = self.models[language]
            tokenizer = self.tokenizers[language]

            inputs = tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output = model(**inputs).waveform

            waveform = output.squeeze().cpu().numpy()
            sample_rate = model.config.sampling_rate

            if output_filename : 
                sf.write(output_filename, waveform, sample_rate)

            return InferenceOutput(waveform=waveform, sample_rate=sample_rate, output_filename=output_filename)

        except Exception as e:
            return InferenceOutput(error_message=f"Error generating audio: {str(e)}")
        


    
if __name__ == "__main__":

    examples = {
        "bambara": "An filɛ ni ye yɔrɔ minna ni an ye an sigi ka a layɛ yala an bɛ ka baara min kɛ ɛsike a kɛlen don ka Ɲɛ wa ?",
        "boomu": "Vunurobe wozomɛ pɛɛ, Poli we zo woro han Deeɓenu wara li Deeɓenu faralo zuun. Lo we baba a lo wara yi see ɓa Zuwifera ma ɓa Gɛrɛkela wa.",
        "dogon": "Pɔɔlɔ, kubɔ lugo joo le, bana dɛin dɛin le, inɛw Ama titiyaanw le digɛu, Ama, emɛ babe bɛrɛ sɔɔ sɔi.",
        "pular": "Miɗo ndaarde saabe Laamɗo e saabe Iisaa Almasiihu caroyoowo wuurɓe e maayɓe oo, miɗo ndaardire saabe gartol makko ka num e Laamu makko",
        "songhoy": "Haya ka se beenediyo kokoyteraydi go hima nda huukoy foo ka fatta ja subaahi ka taasi goykoyyo ngu rezẽ faridi se",
        "tamasheq": "Toḍă tăfukt ɣas, issăɣră-dd măssi-s n-ašĕkrĕš ănaẓraf-net, inn'-as: 'Ǝɣĕr-dd inaxdimăn, tĕẓlĕd-asăn, sănt s-wi dd-ĕšrăynen har tĕkkĕd wi dd-ăzzarnen."
    }

    try:

        ##################################
        #### test all languages
        ##################################

        tts = MalianTTS()
        for lang, text in examples.items():

            output = tts.synthesize(
                text=text,
                language=lang,
                output_filename=f"test_{lang}.wav"
            )

            if output.error_message:
                print({output.error_message})


        ######################################
        ##### test loading specific languages
        ######################################

        tts = MalianTTS(languages=["bambara", "dogon", "tamasheq"]) 
        for lang, text in examples.items():
            output = tts.synthesize(
                text=text,
                language=lang,
                output_filename=f"test_2_{lang}.wav"
            )

            if output.error_message:
                print({output.error_message})


    except Exception as e:
        print(e)





