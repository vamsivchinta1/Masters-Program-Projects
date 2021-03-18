import pymongo  as pm
import pandas as pd
from textblob import TextBlob
import pymongo
import json
import os


pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',100)
pd.set_option('display.width',None)

client = pm.MongoClient('localhost',27017)
db = client.HotelReview
col = db.RatingReviewdata
cursor = db.RatingReviewdata.find()

print(db.HotelReview.show_collections)

data = pd.DataFrame(list(cursor))
data.columns = data.columns.str.replace(' ','')
print(data.head(10))
#comments = pd.DataFrame(data['ReviewText'])
comments = data['ReviewText']
#comments = comments.dropna
#comments = str(comments)
print(comments)

sent=[]
for row in comments:
    text = TextBlob(row)
    snt = text.sentiment.polarity
    sent.append((str(row), snt))
print(sent)
sent1 =pd.DataFrame(sent)
sent1.columns = ['ReviewText','sent']
print(sent1)


left = data.set_index('ReviewText')
right = sent1.set_index('ReviewText')
result = pd.DataFrame(left.join(right))
print(result.head())
result.to_csv('Mongo_Result.csv')

def import_content(filepath):
    mng_client = pymongo.MongoClient('localhost',27017)
    mng_db = mng_client['HotelReview']
    collection_name = 'RatingAnalysis'
    db_cm = mng_db[collection_name]
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)
    
    data_upload = pd.read_csv(file_res)
    data_json = json.loads(data_upload.to_json(orient='records'))
    #db_cm.remove()
    db_cm.drop()
    #db_cm.insert(data_json)
    db_cm.insert_many(data_json)
    
if __name__ == "__main1__":
        filepath = 'C:\\Users\\Noorulain\\Documents\\MSA 8040\\Project\\Mongo_Result.csv'
        import_content(filepath)
        
        


#result1 = pd.DataFrame(result)
# print(result1) 
