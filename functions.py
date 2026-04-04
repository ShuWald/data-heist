import pandas as pd
import numpy as np
import os

class FileData:
    def __init__(self, filename, folderpath='Datasets/'):
        self.filename = filename
        self.folderpath = folderpath
        self.filepath = os.path.join(folderpath, filename)
    
    @property
    def dataframe(self):
        return pd.read_csv(self.filepath)

def textToArray(file):
    with open(file, 'r', encoding='utf-8') as file:
        content = file.read()
    tweets = content.splitlines()
    #print(len(tweets))
    return tweets

tweets = textToArray('Datasets/tweets.txt')
emojis = textToArray('Datasets/emoji.txt')

ConsumerData = FileData('ConsumerData.csv')
FONEData = FileData('FONEData.csv')
USAddressData = FileData('USAddressData.csv')
ZipData = FileData('ZipData.csv')
MigrationData = FileData('OC2025_ZIPMigration.csv')

#print(ConsumerData.shape[0], FONEData.shape[0], USAddressData.shape[0], ZipData.shape[0])

