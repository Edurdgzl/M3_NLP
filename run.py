from ner.ner import plot_error, train_model
from sentiment_analysis.sentiment_analysis import sentiment_analysis
from translation_models.bleu_score import bleu_score
from translation_models.translation_models import translate

print('\n***************** FIRST TASK *****************')

# Call function that prints if the review is POSITIVE or NEGATIVE.
task_one = sentiment_analysis()


print('\n\n***************** SECOND TASK *****************')
print('The plot image is in ner/model/training.png')

# Call unction that trains the model.
task_two_train = train_model()

# Call function that plots train set error and test set error rate during training.
task_two_plot = plot_error()


print('\n\n***************** THIRD TASK *****************')

# Call function that translates a file with Google Translate API and DeepL API.
task_three_transale = translate()

# Call function that prints the bleu score of Google Translator API and DeepL API.
task_three_score = bleu_score()