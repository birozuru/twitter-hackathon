import tweepy
import logging
import os

logger = logging.getLogger()


# To set your environment variables in your terminal run the following line:
# export CONSUMER_KEY='<your_consumer_key>'
# export CONSUMER_SECRET='<your_consumer_secret>'
# export ACCESS_TOKEN='<your_access_token>'
# export ACCESS_TOKEN_SECRET='<your_access_token_secret>'

def create_api():
    consumer_key = "gpWAlpCn0rLjeyzylnp3Tah26"
    consumer_secret = "5A4WsqVpUL0xxATFFV1PXJiX6NOVIb6RIbC9s2s50gPFSEkWCl"
    access_token = "287611088-UXv6J2SvgjUEuKfjEyZIryx6sN0KfDr37GHmy9qy"
    access_token_secret = "Nep5Ysih1im6QFP68iThRFS7SWN4wgmAKA07877jcGklV"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    logger.info("API created")
    return api


