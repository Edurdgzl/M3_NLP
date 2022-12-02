from nltk.translate.bleu_score import sentence_bleu

def bleu_score():

    """Function that opens the english europarl txt file, and the txt files from the results, 
        compares the results with the europarl and prints each api score"""

    europarl = open('translation_models/europarl_texts/europarl-en.txt', 'r')
    europarl_contents = europarl.read()
    europarl_contents = [europarl_contents.split()]

    google_text = open('translation_models/results/google_result.txt', 'r')
    google_contents = google_text.read()
    google_contents = google_contents.split()

    deepl_text = open('translation_models/results/deepl_result.txt', 'r')
    deepl_contents = deepl_text.read()
    deepl_contents = deepl_contents.split()


    print('GOOGLE_TRANSLATOR: {:.2f}'.format(sentence_bleu(europarl_contents, google_contents)))
    print('DEEPL_TRANSLATOR: {:.2f}'.format(sentence_bleu(europarl_contents, deepl_contents)))