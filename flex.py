import tweepy
import config
import connexion


# Setup authentication
auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tweepy.API(auth)


def main():
    store_tweets = []
    tweets = api.search("ganja", count=10)
    # print(tweets)

    for tweet in tweets:
        print(tweet.user.screen_name)


if __name__ == "__main__":
    main()
