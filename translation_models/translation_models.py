from google_cloud_translator import google_cloud_translator
from deepl_translator import deepl_translator

def translate():

    """ Function that that opens the txt file with the europarl,
        saves the content of the file in contents,
        calls google_cloud_translator and deepl_translator functions 
        and saves both results in two different txt files in results folder."""

    f = open('translation_models/europarl_texts/europarl-es.txt', 'r')
    contents = f.read()


    google_result = google_cloud_translator(contents, "en")
    with open('translation_models/results/google_result.txt', 'w') as f:
        f.write(google_result)


    deep_result = deepl_translator(contents, "EN-US")
    with open('translation_models/results/deepl_result.txt', 'w') as f:
        f.write(deep_result)