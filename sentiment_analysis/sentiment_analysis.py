from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download(["names", "stopwords", "movie_reviews", "vader_lexicon", "punkt"]) # do you need this line?

"""
 It’s best practice to have your code in a descriptive method or small class, if possible, rather than running at the top-level.  makes it easier for other modules to import the functionality later if needed! 

"""


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
