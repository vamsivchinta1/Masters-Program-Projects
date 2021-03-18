import os
import pandas as pd
import pymongo
import json


def import_content(filepath):
    mng_client = pymongo.MongoClient('localhost',27017)
    mng_db = mng_client['HotelReview']
    collection_name = 'RatingReviewdata'
    db_cm = mng_db[collection_name]
    cdir = os.path.dirname(__file__)
    file_res = os.path.join(cdir, filepath)
    
    data = pd.read_csv(file_res)
    data_json = json.loads(data.to_json(orient='records'))
    #db_cm.remove()
    db_cm.drop()
    #db_cm.insert(data_json)
    db_cm.insert_many(data_json)
    
if __name__ == "__main__":
        filepath = 'C:\\Users\\Noorulain\\Documents\\MSA 8040\\Project\\Tripadvisor_Dataset_New.csv'
        import_content(filepath)
        
        