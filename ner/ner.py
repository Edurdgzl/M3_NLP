from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.visual.training_curves import Plotter
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.datasets import WNUT_17

def train_model():
    corpus: Corpus = WNUT_17().downsample(0.1)

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)


    embedding_types = [WordEmbeddings('glove')]

    embeddings = StackedEmbeddings(embeddings=embedding_types)


    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type=tag_type,
                            use_crf=True)

    trainer = ModelTrainer(tagger, corpus)

    trainer.train('ner/model',
                learning_rate=0.1,
                mini_batch_size=32,
                max_epochs=150)

def plot_error():
    plotter = Plotter()
    plotter.plot_training_curves('ner/model/loss.tsv')