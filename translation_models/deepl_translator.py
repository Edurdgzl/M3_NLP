import os
import deepl

def deepl_translator(text, target):
    auth_key = os.environ['AUTH_KEY']
    translator = deepl.Translator(auth_key)

    result = translator.translate_text(text, target_lang=target)
    return result.text