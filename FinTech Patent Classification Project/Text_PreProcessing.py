"""
Text Pre-Processing
"""
import warnings
warnings.filterwarnings("ignore")                     	   	  	   	            # Ignoring unnecessory warnings

import numpy as np                                                              # for large and multi-dimensional arrays
import pandas as pd                                                             # for data manipulation and analysis

import nltk                                                                                     # Natural language processing tool-kit
  
                                                               
  
                                            
from nltk.stem import PorterStemmer                                             # Stemmer


from sklearn.feature_extraction.text import CountVectorizer                     #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer                     #For TF-IDF
from gensim.models import Word2Vec                                              #For Word2Vec
#%% Directory
import os

dp = 'C:/Users/vamsi/Desktop/Project_A'
os.chdir(dp)
os.getcwd()
#%% Importing Data
df = pd.read_csv('patent_records_train.csv',\
				 encoding = "ISO-8859-1" )
df.head(10)  
#%% Stopwords
from nltk.corpus import stopwords                                                              # Stopwords corpus
nltk.download('stopwords') 

stop = set(stopwords.words('english'))
#%%  Text PreProcessing
#	converting to lowercase/remving punctuation/stemming/removing stopwords
import re

df_txt 		= df['Text']
temp 		= []
x 			= nltk.stem.SnowballStemmer('english')

for sentence in df_txt:
    sentence 	= sentence.lower()                 	                                          # Converting to lowercase
    # cleanr 	= re.compile('<.*?>')
    # sentence 	= re.sub(cleanr, ' ', sentence)                                             #Removing HTML tags
    sentence 	= re.sub(r'[?|!|\'|"|#]',\
					r'',\
					sentence)
    sentence 	= re.sub(r'[.|,|)|(|\|/]',\
					r' ',\
					sentence)                                                              #Removing Punctuations
    
    words 	= [x.stem(word) for word in sentence.split()\
			  if word not in stopwords.words('english')]   	                                 # Stemming and removing stopwords
	
    temp.append(words)
    
df_txt 		= temp    
print(df_txt[1])

#%% compile transformed text data
x2 			= []
for row in df_txt:
    sequ 			= ''
    for word in row:
        sequ 		= sequ + ' ' + word
    x2.append(sequ)

df_txt 		= x2
print(df_txt[1])

#%%


