# -*- coding: utf-8 -*-
"""Nueral Network Recommendation System - Data Exploration.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cJ6SWzUFp62HC2em638Kp7izTHhjd68_

# Hw3
### Data Exploration

---



---
"""

from google.colab import drive
drive.mount('/content/gdrive')

import os
path = r'/content/gdrive/My Drive/Colab_Datasets'
os.chdir(path)

!pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
!pip install git+https://github.com/maciejkula/spotlight.git@master#egg=spotlight

# Commented out IPython magic to ensure Python compatibility.
import gzip
import json

import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

df = pd.read_csv('DigitalMusic.csv')
df.head()

# Ratings
# cnt   = ~1.6mm ratings
# avg   = ~3.6 +/- 1.6
# Products
# cntd  = ~460,000 products
# avg   = ~20 +/- 140 ratings per product
#       = ~3.3 +/- 1.3 ratings per user
# Users
# cntd  = ~840,000 users
# avg   = ~1 +/- 1 ratings per user
#       = ~3.5 +/- 1.6 ratings per product

# df.describe()
# df['SoftwareID'].nunique()
# df['UserID'].nunique()
# df[['SoftwareID', 'Rating']].groupby(['SoftwareID']).agg(['count','mean']).mean()
# df[['SoftwareID', 'Rating']].groupby(['SoftwareID']).agg(['count','mean']).std()
# df[['UserID', 'Rating']].groupby(['UserID']).agg(['count','mean']).mean()
# df[['UserID', 'Rating']].groupby(['UserID']).agg(['count','mean']).std()

print(df.columns.values.tolist())

df['asin'].nunique()

df['asin'].count()

df['reviewerID'].nunique()

ratings = pd.DataFrame(df.groupby('asin')['overall'].mean())
ratings.head()
ratings['number_of_ratings'] = df.groupby('asin')['overall'].count()
ratings.head()

plt.figure(figsize=[80,40])

ratings['number_of_ratings'].hist(bins=10000)
plt.xlim(xmin=0, xmax = 10)
plt.title('Review Count Distribution',fontsize=150)
plt.xlabel('Total # of Reviews',fontsize=70)
plt.ylabel('# of Products',fontsize=70)
plt.show()

# Joint Plot
fontsize = 10

fig, axes = plt.subplots()
fig, axes = plt.subplots(figsize=(20,10))

sns.jointplot(x='rating', y='number_of_ratings', data=ratings)

axes.set_title('User Ratings Distribution Plot', fontsize=50)

axes.yaxis.grid(True)
axes.set_xlabel('Ratings',fontsize=25)
axes.set_ylabel('Count',fontsize=25)

plt.show();

# Violin Plots
fontsize = 10

fig, axes = plt.subplots()
fig, axes = plt.subplots(figsize=(20,10))

axes= sns.violinplot(dataset = df,
                     x = df['overall'], 
                     scale = "width")

axes.set_title('User Ratings Distribution Plot', fontsize=50)

axes.yaxis.grid(True)
axes.set_xlabel('Ratings',fontsize=25)
axes.set_ylabel('Count',fontsize=25)

plt.show();
