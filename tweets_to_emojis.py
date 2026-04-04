from functions import textToArray
import pandas as pd

texts = textToArray('Datasets/tweets.txt')
emojis = textToArray('Datasets/emoji.txt')

df = pd.DataFrame({'texts': texts, 'emojis': emojis})

print(df)