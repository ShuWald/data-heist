import pandas as pd
import numpy as np

def textToArray(file):
    with open(file, 'r', encoding='utf-8') as file:
        content = file.read()
    tweets = content.splitlines()
    #print(len(tweets))
    return tweets

tweets = textToArray('Datasets/tweets.txt')
emojis = textToArray('Datasets/emoji.txt')

data_files = ['Datasets/ConsumerData.csv', 'Datasets/FONEData.csv', 'Datasets/USAddressData.csv', 'Datasets/ZipData.csv']
ConsumerData, FONEData, USAddressData, ZipData = [pd.read_csv(file) for file in data_files]
#print(ConsumerData.shape[0], FONEData.shape[0], USAddressData.shape[0], ZipData.shape[0])

useful_tweets = pd.read_csv('Datasets/tweets_formatted.csv')
details = useful_tweets.columns


