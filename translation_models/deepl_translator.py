import os
import deepl


def deepl_translator(text, target):

    """Function that receives the text file path and the target language.
        Reads the api key. Creates the translator object with the key, 
        translates the text and returns the translated text."""

    auth_key = os.environ['AUTH_KEY']
    translator = deepl.Translator(auth_key)

    result = translator.translate_text(text, target_lang=target)
    return result.text