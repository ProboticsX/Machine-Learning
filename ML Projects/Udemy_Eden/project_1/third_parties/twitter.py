import os
from dotenv import load_dotenv
import requests
import tweepy

load_dotenv()
twitter_client = tweepy.Client(
    bearer_token=os.environ["TWITTER_BEARER_TOKEN"],
    consumer_key=os.environ["TWITTER_API_KEY"],
    consumer_secret=os.environ["TWITTER_API_KEY_SECRET"],
    access_token=os.environ["TWITTER_ACCESS_TOKEN"],
    access_token_secret=os.environ["TWITTER_ACCESS_TOKEN_SECRET"],
)

def scrape_user_tweets(username, num_tweets=5, mock:bool=False):
    '''
        Scrapes a user's original tweets and not retweets or replies and returns them as a list of dictionaries
        Each dictionary should have 3 fields: "time posted"(relative to now), "text" and "url"
    '''
    tweet_list = []
    if mock:
        twitter_gist = "https://gist.githubusercontent.com/ProboticsX/f49ab683745913c5f6c9e053ad1782b4/raw/24fa8d232fa88fa599c3ea630c5705e6139f523f/elonmusk-tweets.json"
        tweets = requests.get(twitter_gist, timeout=5).json()
        for tweet in tweets:
            tweet_dict = {}
            tweet_dict["text"] = tweet["text"]
            tweet_dict["url"] = tweet["url"]
            tweet_list.append(tweet_dict)
    else:
        user_id = twitter_client.get_user(username=username).data.id
        tweets = twitter_client.get_users_tweets(id=user_id, max_results=num_tweets, exclude=["retweets", "replies"])
        for tweet in tweets.data:
            tweet_dict = {}
            tweet_dict["text"] = tweet["text"]
            tweet_dict["url"] = f"https://twitter.com/{username}/status/{tweet.id}"
            tweet_list.append(tweet_dict)
    return tweet_list

if __name__ == "__main__":

    tweets = scrape_user_tweets(username="elonmusk", mock=True)
    print(tweets)
