import os
from google.cloud import language_v1

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.curdir, 'twitter-hack-296312-1687853aa8bb.json')

def get_sentiment_score(tweet):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=tweet, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
    sentiment_score = sentiment.score
    sentiment_magnitude = sentiment.magnitude

    print(sentiment_score, sentiment_magnitude)

tweets = ['This product really sucks i hate the new update', 'This new twitter fleets is so annoying gosh', 'I am going to throw this new iphones away  cause i do not fancy the architecture']

for tweet in tweets:
    get_sentiment_score(tweet)