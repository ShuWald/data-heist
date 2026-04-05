from functions import FileData
import pandas as pd

useful_data = FileData('ConsumerData.csv')

df = useful_data.dataframe

df.loc[:, 'Address'] = df.loc[:, 'Address'].apply(lambda address: address[address.find(' ') + 1:])

useless_columns = ['RecordID', 'MAK', 'BaseMak', 'City', 'State', 'Zipcode', 'Latitude', 'Longitude']
useful_data = df.drop(columns=useless_columns)
details = useful_data.columns

print(details)
print(useful_data.to_numpy().flatten())

# useful_data.to_csv('clean_consumer_data.csv', index = False)