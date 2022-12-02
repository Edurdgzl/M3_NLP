import os
from google.cloud import translate_v2 as translate

# Read api key.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"credential_route"


def google_cloud_translator(text, target):

    """Function that receives the text file path and the target language.
        Creates the client object, translates the text and returns the translated text."""

    translate_client = translate.Client()

    result = translate_client.translate(text, target_language=target)
    return result["translatedText"]