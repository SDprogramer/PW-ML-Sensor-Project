from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# url
uri = "mongodb+srv://sd:<db_password>@cluster0.mw4ir.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# create a new client and connectt to server
client = MongoClient(uri)

# create database name and collection name
MONGO_DATABASE_NAME = "sensor_project"
MONGO_COLLECTION_NAME = "waferfault"

df = pd.read_csv("C:\PROGRAMING\DS ML Gen AI\ML Sensor Project\notebooks\wafer_23012020_041211.csv")

df = df.drop("Unnamed: 0", axis = 1)

json_record = list(json.loads(df.T.to_json()).values())

client[MONGO_DATABASE_NAME][MONGO_COLLECTION_NAME].insert_many(json_record)
