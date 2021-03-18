import pandas as pd
import numpy as np
from sklearn import preprocessing
from pprint import pprint


# csv file -> dataframe
df = pd.read_csv('wineData.csv')

# class map target features
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['Class']))}
tempCol = df.pop('Class')

# normalized dataframe
min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(0, 2))
df_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
df_scaled['Class'] = tempCol
df_scaled['Class'] = df_scaled['Class'].map(class_mapping)
pprint(df_scaled)

# dataframe -> csv file
df_scaled.to_csv('wineNormalized.csv', index=False)
