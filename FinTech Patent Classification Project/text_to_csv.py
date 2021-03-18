"""
converts 'train' dataset from text to a formatted csv file
"""
import re
import pandas as pd
#%%

with open("train.txt","r") as file:
	text = file.read()
	my_list = []
	cols = re.split('(\d{11})    ', text)[1:]

	for i in range(0, len(cols)-1, 2):
		my_list.append((cols[i], cols[i+1]))
	fieldnames = ['Patent_Number', 'Text']
	df = pd.DataFrame(my_list, columns=fieldnames)
	df.to_excel('patent_records_train.xlsx', index=False)
#%%