from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download(["names", "stopwords", "movie_reviews", "vader_lexicon", "punkt"])
sia = SentimentIntensityAnalyzer()
def sentiment_analysis():
    with open('sentiment_analysis/dataset/tiny_movie_reviews_dataset.txt') as f:
        lines = f.readlines()
        for line in lines:
            if not line: continue 
            sentiment = sia.polarity_scores(line)
            if (sentiment['neg'] > sentiment['pos']):
                print('NEGATIVE')
            else: # dont need the extra conditional
                print('POSITIVE')