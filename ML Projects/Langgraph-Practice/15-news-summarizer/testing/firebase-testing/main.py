from dotenv import load_dotenv
import os
load_dotenv()
import firebase_admin
from firebase_admin import credentials, firestore
import time
import json
from datetime import datetime

# Path to your service account key
cred = credentials.Certificate("./firebaseServiceAccountKey.json")
firebase_admin.initialize_app(cred)

# Get a reference to Firestore
db = firestore.client()

# Read the JSON file
json_file_path = "../../data/summary/top_headlines_summary.json"
with open(json_file_path, 'r') as file:
    headlines = json.load(file)

# Get today's date in YYYY-MM-DD format
today_date = datetime.now().strftime("%Y-%m-%d")

# Reference to today's headlines collection
today_headlines_ref = db.collection("news").document("top_headlines").collection(today_date)

# Check if there are existing headlines and delete them
existing_headlines = today_headlines_ref.get()
if existing_headlines:
    print(f"Found existing headlines for {today_date}, deleting them...")
    for doc in existing_headlines:
        doc.reference.delete()
    print("Existing headlines deleted")

# Push new data to Firestore
for index, headline in enumerate(headlines):
    # Create a document reference with today's date and index
    doc_ref = today_headlines_ref.document(f"headline_{index}")
    
    # Add timestamp to the data
    headline_data = {
        **headline,
        "timestamp": firestore.SERVER_TIMESTAMP
    }
    
    # Save to Firestore
    doc_ref.set(headline_data)
    print(f"Pushed headline {index + 1} to Firestore")

print("All headlines have been pushed to Firestore")
