from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download(["names", "stopwords", "movie_reviews", "vader_lexicon", "punkt"])
sia = SentimentIntensityAnalyzer()
def sentiment_analysis():
    with open('sentiment_analysis/dataset/tiny_movie_reviews_dataset.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            sentiment = sia.polarity_scores(line)
            if (sentiment['neg'] > sentiment['pos']):
                print('NEGATIVE')
            elif (sentiment['neg'] < sentiment['pos']):
                print('POSITIVE')