import tweepy
import logging
import flask
import connexion
import timing
import config
import re
from nltk.tokenize import WordPunctTokenizer
import Gfunc

logging.basicConfig(filename='bot.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s %(funcName)s:%(msecs)d',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()

# Create the application instance
app = connexion.App(__name__, specification_dir='./')
api = config.create_api()

# Read the swagger.yml file to configure the endpoints
#app.add_api('swagger.yml')


# Create a URL route in our application for "/"
@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/
    :return:        the rendered template 'home.html'
    """
    return "<h1>Hello World</h1>"


def  search_tweets(keyword, total_tweets):
    timing.today_datetime
    timing.yesterday_datetime
    timing.today_date
    timing.yesterday_date
    search_result = tweepy.Cursor(api.search,
                    q=keyword,
                    since=timing.yesterday_date,
                    result_type='recent',
                    lang='en').items(total_tweets)

    return search_result

def clean_tweets(tweet):
    user_removed = re.sub(r'@[A-Za-z0-9]+','',tweet.decode('utf-8'))
    link_removed = re.sub('https?://[A-Za-z0-9./]+','',user_removed)
    number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
    lower_case_tweet= number_removed.lower()
    tok = WordPunctTokenizer()
    words = tok.tokenize(lower_case_tweet)
    clean_tweet = (' '.join(words)).strip()
    return clean_tweet


def analyze_tweets(keyword, total_tweets):
    score = 0
    tweets = search_tweets(keyword, total_tweets)
    for tweet in tweets:
        cleaned_tweet = clean_tweets(tweet.text.encode('utf-8'))
        sentiment_score = Gfunc.get_sentiment_score(cleaned_tweet)
        score += sentiment_score
        print('Tweet: {}'.format(cleaned_tweet))
        print('Score: {}\n'.format(sentiment_score))
    final_score = round((score / float(total_tweets)),2)
    return final_score



# If we're running in stand alone mode, run the application
if __name__ == '__main__':
    analyze_tweets("asuu", 50)