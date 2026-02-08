# library used by test

# class to test
# nllb translator
from maliba_ai.core.mt.models.nllb import BambaraTranslator
from maliba_ai.settings.mt.nllb import Language, Languages
from maliba_ai.settings.mt.settings import TranslationOutput
# To generate data samples
from maliba_ai.tests.inference_data_samples import inference_data_samples


# nllb Bambara Translator
def test_nllb_bambara_translator(
    src_lang: Language, tgt_lang: Language, test_data_samples: list[str]
):
    translator = BambaraTranslator()

    # test 1 - no text provided
    # result: TranslationOutput object with error message
    output = translator.translate("", src_lang, tgt_lang)
    assert isinstance(output, TranslationOutput)
    assert output.translation is None
    assert output.error_message == "text cannot be empty"

    # test 2 - source language and target language are the same
    # result: TranslationOutput object with error message
    for data_sample in inference_data_samples("MT"):
        output = translator.translate(data_sample, src_lang, src_lang)
        assert isinstance(output, TranslationOutput)
        assert output.translation is None
        assert output.error_message == "source and target languages must be different"

    # test 3 - successful translation
    # result: TranslationOutput object with no error message
    for data_sample in test_data_samples("MT"):
        output = translator.translate(data_sample, src_lang, tgt_lang)
        # Check for correct return type
        assert isinstance(output, TranslationOutput)
        assert isinstance(output.translation, str) or (
            isinstance(output.translation, list)
            and all(isinstance(item, str) for item in output.translation)
        )
        assert output.error_message is None


if __name__ == "__main__":
    # defining language objects for the test
    languages = Languages()
    bambara = languages.bambara
    french = languages.french
    english = languages.english
    test_nllb_bambara_translator(bambara, french, inference_data_samples("MT"))
    test_nllb_bambara_translator(french, bambara, inference_data_samples("MT"))
    test_nllb_bambara_translator(english, bambara, inference_data_samples("MT"))
    test_nllb_bambara_translator(bambara, english, inference_data_samples("MT"))
