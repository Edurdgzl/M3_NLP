import os
from google.cloud import translate_v2 as translate

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"credential_route"

def google_cloud_translator(text, target):
    translate_client = translate.Client()

    result = translate_client.translate(text, target_language=target)
    return result["translatedText"]