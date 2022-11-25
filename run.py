from ner.ner import plot_error, train_model
from sentiment_analysis.sentiment_analysis import sentiment_analysis
from translation_models.bleu_score import bleu_score
from translation_models.translation_models import translate

print('\n***************** FIRST TASK *****************')
task_one = sentiment_analysis()


print('\n\n***************** SECOND TASK *****************')
print('The plot image is in ner/model/training.png')
task_two_train = train_model()
task_two_plot = plot_error()


print('\n\n***************** THIRD TASK *****************')
task_three_transale = translate()
task_three_score = bleu_score()