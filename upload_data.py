## Do whole code chatgpt

from pymongo.mongo_client import MongoClient
import pandas as pd
import json

## url
uri  = "mongodb+srv://sd:<your-password>@cluster0.pff51.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

## Create a new client and connect to server
client = MongoClient(uri)

## Create database name and collection time
DATABASE_NAME = "PwSkills"
COLLECTION_NAME = "waferfault"

df = pd.read_csv(r"E:\PROGRAMING\DS ML Gen AI\ML Sensor Project\notebooks\wafer_23012020_041211.csv")
df = df.drop("Unnamed: 0", axis = 1)

json_record = list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)