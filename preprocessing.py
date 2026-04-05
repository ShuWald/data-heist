from functions import FileData
import pandas as pd
import sklearn

useful_data = FileData('ConsumerData.csv')

df = useful_data.dataframe

df.loc[:, 'Address'] = df.loc[:, 'Address'].apply(lambda address: address[address.find(' ') + 1:])

useless_columns = ['RecordID', 'MAK', 'BaseMak', 'City', 'State', 'Zipcode', 'Latitude', 'Longitude']
useful_data = df.drop(columns=useless_columns)
details = useful_data.columns

print(details)
print(useful_data.to_numpy().flatten())

X = df.drop(columns=['Address'])
y = df['Address']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# useful_data.to_csv('Datasets/clean_consumer_data.csv', index = False)