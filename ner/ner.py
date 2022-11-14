import urllib.request
from pathlib import Path
from flair.data import Corpus
from flair.datasets import ColumnCorpus
import pandas as pd
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings


def download_file(url, output_file):
  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  urllib.request.urlretrieve (url, output_file)

""" download_file('https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_train.txt', 'ner/data/conllpp_train.txt')
download_file('https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_dev.txt', 'ner/data/conllpp_dev.txt')
download_file('https://raw.githubusercontent.com/ZihanWangKi/CrossWeigh/master/data/conllpp_test.txt', 'ner/data/conllpp_test.txt') """


columns = {0: 'text', 3: 'ner'}
corpus: Corpus = ColumnCorpus('ner/data/', columns,
                              train_file='conllpp_train.txt',
                              test_file='conllpp_test.txt',
                              dev_file='conllpp_dev.txt')


data = [[len(corpus.train), len(corpus.test), len(corpus.dev)]]
# Prints out the dataset sizes of train test and development in a table.
pd.DataFrame(data, columns=["Train", "Test", "Development"])


tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)


embedding_types = [WordEmbeddings('glove'), FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')]

embeddings = StackedEmbeddings(embeddings=embedding_types)


# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type='ner',
                        use_crf=False,
                        use_rnn=False,
                        reproject_embeddings=False,
                        )

trainer = ModelTrainer(tagger, corpus)

trainer.train('ner/model/conllpp',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=50,
              embeddings_storage_mode='gpu')