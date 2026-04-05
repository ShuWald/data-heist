import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


"""
Calculate regression metrics comparing predictions to expected values.
Input a model_name to print.
"""
def calculate_metrics(y_true, y_pred, model_name=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.any(y_true != 0) else np.inf
    
    metrics = {
        'MAE': mae,           # Mean Absolute Error
        'MSE': mse,           # Mean Squared Error
        'RMSE': rmse,         # Root Mean Squared Error
        'R2': r2,             # R² Score (1.0 is perfect)
        'MAPE': mape          # Mean Absolute Percentage Error
    }
    
    if model_name is not None:
        print(f"\n{'='*40}")
        print(f"Metrics for {model_name}")
        print(f"{'='*40}")
        print(f"R² Score:        {metrics['R2']:.6f}  (1.0 = perfect, 0.0 = baseline)")
        print(f"MAE:             {metrics['MAE']:.6f}  (Mean Absolute Error)")
        print(f"RMSE:            {metrics['RMSE']:.6f}  (Root Mean Squared Error)")
        print(f"MAPE:            {metrics['MAPE']:.2f}%  (Mean Absolute % Error)")
        print(f"{'='*40}")
    
    return metrics

'''
tweets = textToArray('Datasets/tweets.txt')
emojis = textToArray('Datasets/emoji.txt')
ConsumerData = FileData('ConsumerData.csv')
FONEData = FileData('FONEData.csv')
USAddressData = FileData('USAddressData.csv')
ZipData = FileData('ZipData.csv')
MigrationData = FileData('OC2025_ZIPMigration.csv')
#print(ConsumerData.shape[0], FONEData.shape[0], USAddressData.shape[0], ZipData.shape[0])
'''
