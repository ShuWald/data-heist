from functions import FileData
import pandas as pd

useful_data = FileData('ConsumerData.csv')
useless_columns = ['RecordID', 'MAK', 'BaseMak', 'Address', 'City', 'State', 'Zipcode']
useful_data = useful_data.dataframe.drop(columns=useless_columns)
details = useful_data.columns
print(details)
print(useful_data.to_numpy().flatten())