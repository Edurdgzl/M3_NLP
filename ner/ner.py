from flair.data import Corpus        
from flair.datasets import CONLL_03
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.data import Sentence
from flair.models import SequenceTagger


corpus: Corpus = CONLL_03()

tag_type = 'ner'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding_types = [
    # GloVe embeddings
    WordEmbeddings('glove'),
    # contextual string embeddings, forward
    FlairEmbeddings('news-forward'),
    # contextual string embeddings, backward
    FlairEmbeddings('news-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)


tagger = SequenceTagger(hidden_size=256,             
                    embeddings=embeddings, 
                    tag_dictionary=tag_dictionary,
                    tag_type=tag_type)


trainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/ner-english',
            train_with_dev=True,
            max_epochs=150)

""" model = SequenceTagger.load('resources/taggers/ner-english/final-model.pt')
sentence = Sentence("George Washington lives in Washington")
model.predict(sentence)
for entity in sentence.get_spans('ner'):
    print(entity) """