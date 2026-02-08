def pytest_addoption(parser):
    parser.addoption(
        "--model", action="store", default="baseline", help="Which AI model to test"
    )

    parser.addoption(
        "--modeltype",
        action="store",
        default="all",
        help="what type of model to test : MT, TTS, ASR, LLM",
    )

    parser.addoption(
        "--srclang",
        action="store",
        default="french",
        help="what language is the original text in: bambara, french, english",
    )

    parser.addoption(
        "--destlang",
        action="store",
        default="bambara",
        help="what lanuage to translate to: bambara, french, english",
    )
