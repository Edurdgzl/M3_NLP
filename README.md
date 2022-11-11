# Module 3 NLP Module Project

Simple overview of use/purpose.

## Description

This project is divided in three sections. The first one is Sentiment Analysis. In this folder there is a folder with the dataset and the file sentiment_analysis.py which uses SentimentIntensityAnalyzer from nltk to do sentiment analysis on 'tiny_movie_reviews_dataset.txt'. The program prints if the review is POSITIVE or NEGATIVE.

The second section is about Named Entity Recognition.

The third section is about translation models. The folder contains a folder with the two versions of the europarl text (English and Spanish, with 100 lines), a folder with two txt files with the results of each model, the deepl_translator.py file which has a function to use the DeepL API, the google_cloud_translator.py file wich has a function to use the Google Cloud Translator API, the translation_mdeols.py which uses the two functions mentioned above with the Spanish version of the europarl text and creates the txt files with the results, and the bleu_score.py which has a function to print the Bleu score of each model.

### Installing

To install the requirements first create a virtual environment:
For Windows use the command 'py -3 -m venv venv', then activate the corresponding environment with the command 'venv\Scripts\activate'
For macOS/Linux use the command 'python3 -m venv venv', then activate the corresponding environment with the command '. venv/bin/activate'

API Keys:
For DeepL put the key in an env file as AUTH_KEY
For Google Cloud Translator put the path of the json file in an env file as GOOGLE_APPLICATION_CREDENTIALS

Finally install the requirements with the command 'pip install -r requirements.txt'

### Executing program

Tests:

To run tests, from the root dir of the repo, call:

```
python3 run.py
```

## Authors

ex. Eduardo Rodríguez López A01749381@tec.mx
