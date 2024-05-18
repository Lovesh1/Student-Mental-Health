# %%
import pandas as pd
import numpy as np


# %%
df = pd.read_csv('./heart_2020_cleaned.csv')

# %%
df.head()

# %%
df.isnull().sum()

# %%
df.duplicated().value_counts()

# %%
df.info()

# %%
df.drop_duplicates(inplace=True)

# %%
df.duplicated().value_counts()

# %%
df['PhysicalHealth'].value_counts()

# %%
df.describe()

# %%
df['SleepTime'].unique()

# %%
df['SleepTime'].value_counts()


# %%
df['BMI'].value_counts()


# %%
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# %%
le = LabelEncoder()
df['HeartDisease']= le.fit_transform(df['HeartDisease'])
df['HeartDisease']

# %%
df.head()

# %%
df = df[df['BMI']<40]
df.describe()

# %%
df.replace({'Yes': 1, 'No' :0, 'Female':0, 'Male':1, 'Poor':0,'Fair':1, 'Good':2, 'Very good': 3, 'Excellent':4},inplace=True)
df

# %%
df['AgeCategory'].value_counts()
import seaborn as sns
import matplotlib.pyplot as plt

df['AgeCategory'].astype(str)
df['AgeCategory']= df['AgeCategory'].str.slice(0,2,1)
df

# %%
df = df[df['SleepTime']<14]

# %%



