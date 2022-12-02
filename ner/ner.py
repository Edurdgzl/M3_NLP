from flair.data import Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.visual.training_curves import Plotter
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.datasets import WNUT_17


def train_model(percent_of_dataset_to_train=0.1):

    """ Function that receives percent of dataset to train (0.1 is recommended). Uses corpus dataset, 
        glove for embedding, a learning rate of 0.1, mini batch of 32 and 150 epochs max. Saves the loss, test and weights in model folder."""

    corpus: Corpus = WNUT_17().downsample(percent_of_dataset_to_train)

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

    """ Function that plots train set error and test set error rate during training. Saves the image in model folder."""

    plotter = Plotter()
    plotter.plot_training_curves('ner/model/loss.tsv')