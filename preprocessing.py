from functions import FileData
import pandas as pd

useful_tweets = FileData('tweets_formatted.csv')
useless_columns = ['tweet_id', 'author_id', 'author_verified', 'author_description']
useful_tweets = useful_tweets.dataframe.drop(columns=useless_columns)
ut_details = useful_tweets.columns
print(ut_details)