from functions import FileData
import pandas as pd
import sklearn
import numpy as np

useful_data = FileData('ConsumerData.csv')

df = useful_data.dataframe

df.loc[:, 'Address'] = df.loc[:, 'Address'].apply(lambda address: address[address.find(' ') + 1:])

# dropping address for now, may use it later with encoding
useless_columns = ['RecordID', 'MAK', 'BaseMak', 'Address', 'City', 'State', 'Zipcode']
df = df.drop(columns=useless_columns)
details = df.columns

print(details)
print(df.to_numpy().flatten())

# y, m, o = 1 (yes, married, owner)
# n, s, r, nan = 0 (no, single, renter)

string_cols = df.select_dtypes(include='object').columns

for col in string_cols:
    df[col] = df[col].apply(lambda val: int(val in ['Y', 'M', 'O'])).astype(int)

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

df.drop(['HomePurchaseDate', 'NumberOfChildren', 'HouseholdSize', 'NetWorth', 'VehicleKnownOwnedNumber'], axis = 1, inplace = True)

X = df.drop(columns=['Latitude', 'Longitude'])
y = pd.concat([df['Latitude'], df['Longitude']], axis = 1)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

print(X_train)
print(y_train)

# useful_data.to_csv('Datasets/clean_consumer_data.csv', index = False)