from typing import List, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import json
import os
from pathlib import Path

class FirebaseTools:
    def __init__(self, service_account_path: str = None):
        """
        Initialize Firebase connection.
        
        Args:
            service_account_path (str, optional): Path to Firebase service account JSON file.
                If not provided, will look for it in the current directory.
        """
        if not service_account_path:
            # Try to find the service account file in the current directory
            current_dir = Path(__file__).parent
            service_account_path = str(current_dir  / "firebaseServiceAccountKey.json")
        
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(f"Firebase service account file not found at: {service_account_path}")
        
        # Initialize Firebase if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
    
    def push_headlines(self, headlines_data: List[Dict[str, Any]], collection_path: str = "news") -> Dict[str, Any]:
        """
        Push headlines to Firestore database.
        
        Args:
            headlines_data (List[Dict[str, Any]]): List of headline dictionaries to push
            collection_path (str): Base collection path where headlines should be stored
            
        Returns:
            Dict[str, Any]: Status of the operation including success/failure and details
        """
        try:
            # Get today's date in YYYY-MM-DD format
            today_date = datetime.now().strftime("%Y-%m-%d")
            
            # Reference to today's headlines collection
            # Structure: news (collection) -> top_headlines (document) -> headlines (collection) -> today's date (document) -> headlines (collection)
            headlines_ref = self.db.collection(collection_path).document("top_headlines").collection("headlines").document(today_date).collection("headlines")
            
            # Check if there are existing headlines and delete them
            existing_headlines = headlines_ref.get()
            if existing_headlines:
                print(f"Found existing headlines for {today_date}, deleting them...")
                # Delete all documents in the collection
                for doc in existing_headlines:
                    doc.reference.delete()
                print("Existing headlines deleted")
            
            # Create a batch for atomic operations
            batch = self.db.batch()
            
            # Add each headline to the batch
            for index, headline in enumerate(headlines_data):
                doc_ref = headlines_ref.document(f"headline_{index}")
                headline_data = {
                    **headline,
                    "timestamp": firestore.SERVER_TIMESTAMP
                }
                batch.set(doc_ref, headline_data)
            
            # Commit the batch
            batch.commit()
            
            return {
                "success": True,
                "message": f"Successfully pushed {len(headlines_data)} headlines to Firestore",
                "date": today_date,
                "headlines_count": len(headlines_data)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error pushing headlines to Firestore: {str(e)}",
                "error": str(e)
            }
    
    def push_headlines_from_json(self, json_file_path: str, collection_path: str = "news") -> Dict[str, Any]:
        """
        Push headlines from a JSON file to Firestore database.
        
        Args:
            json_file_path (str): Path to the JSON file containing headlines
            collection_path (str): Base collection path where headlines should be stored
            
        Returns:
            Dict[str, Any]: Status of the operation including success/failure and details
        """
        try:
            # Read the JSON file
            with open(json_file_path, 'r') as file:
                headlines = json.load(file)
            
            # Push the headlines
            return self.push_headlines(headlines, collection_path)
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error reading or processing JSON file: {str(e)}",
                "error": str(e)
            }

# Example usage:
if __name__ == "__main__":
    # Initialize the Firebase tools
    firebase_tools = FirebaseTools()
    
    # Example of pushing headlines from a JSON file
    json_path = "../../data/summary/top_headlines_summary.json"
    result = firebase_tools.push_headlines_from_json(json_path)
    print(result)