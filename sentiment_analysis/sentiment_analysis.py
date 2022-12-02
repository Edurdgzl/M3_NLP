from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download data.
nltk.download(["names", "stopwords", "movie_reviews", "vader_lexicon", "punkt"])

# Create SentimentIntensityAnalyzer.
sia = SentimentIntensityAnalyzer()


def sentiment_analysis():

    """ Function that opens the txt file with the reviews, 
        reads the lines of the txt file and prints if it is POSITIVE or NEGATIVE."""

    with open('sentiment_analysis/dataset/tiny_movie_reviews_dataset.txt') as f:
        lines = f.readlines()
        for line in lines:
            if not line: continue 
            sentiment = sia.polarity_scores(line)
            if (sentiment['neg'] > sentiment['pos']):
                print('NEGATIVE')
            else:
                print('POSITIVE')