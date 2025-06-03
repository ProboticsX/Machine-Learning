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
    
    def _convert_timestamp_to_iso(self, data: Any) -> Any:
        """
        Convert Firestore timestamps to ISO format strings recursively.
        
        Args:
            data: The data to convert (can be dict, list, or primitive type)
            
        Returns:
            The converted data with timestamps as ISO strings
        """
        if isinstance(data, dict):
            return {k: self._convert_timestamp_to_iso(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_timestamp_to_iso(item) for item in data]
        elif hasattr(data, 'timestamp'):  # Check if it's a Firestore timestamp
            return data.isoformat()
        return data

    def push_headlines(self, headlines_data: List[Dict[str, Any]], collection_path: str = "news_latest", category: str = "general") -> Dict[str, Any]:
        """
        Push headlines to Firestore database.
        
        Args:
            headlines_data (List[Dict[str, Any]]): List of headline dictionaries to push
            collection_path (str): Base collection path where headlines should be stored
            category (str): Category of the headlines (e.g., technology, business, general)
            
        Returns:
            Dict[str, Any]: Status of the operation including success/failure and details
        """
        try:
            # Get today's date in YYYY-MM-DD format
            today_date = datetime.now().strftime("%Y-%m-%d")
            
            # Reference to today's headlines collection
            # Structure: news_latest (collection) -> top_headlines (document) -> category (document) -> date (document) -> headlines (collection) -> headline_num (documents)
            headlines_collection = (
                self.db.collection(collection_path)
                .document("top_headlines")
                .collection("categories")
                .document(category)
                .collection("dates")
                .document(today_date)
                .collection("headlines")
            )
            
            # Check if there are existing headlines and delete them
            existing_docs = headlines_collection.get()
            if existing_docs:
                print(f"Found existing headlines for {category} on {today_date}, deleting them...")
                batch = self.db.batch()
                for doc in existing_docs:
                    batch.delete(doc.reference)
                batch.commit()
                print(f"Deleted existing headlines for {category} on {today_date}")
            
            # Create a batch for atomic operations
            batch = self.db.batch()
            
            # Add each headline as a separate document
            for index, headline in enumerate(headlines_data):
                doc_ref = headlines_collection.document(f"headline_{index}")
                headline_data = {
                    **headline,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "category": category,
                    "date": today_date,
                    "index": index
                }
                batch.set(doc_ref, headline_data)
            
            # Commit the batch
            batch.commit()
            
            return {
                "success": True,
                "message": f"Successfully pushed {len(headlines_data)} headlines to Firestore for category: {category}",
                "date": today_date,
                "category": category,
                "headlines_count": len(headlines_data)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error pushing headlines to Firestore: {str(e)}",
                "error": str(e)
            }
    
    def push_headlines_from_json(self, json_file_path: str, collection_path: str = "news_latest", category: str = "general") -> Dict[str, Any]:
        """
        Push headlines from a JSON file to Firestore database.
        
        Args:
            json_file_path (str): Path to the JSON file containing headlines
            collection_path (str): Base collection path where headlines should be stored
            category (str): Category of the headlines (e.g., technology, business, general)
            
        Returns:
            Dict[str, Any]: Status of the operation including success/failure and details
        """
        try:
            # Read the JSON file
            with open(json_file_path, 'r') as file:
                headlines = json.load(file)
            
            # Push the headlines
            return self.push_headlines(headlines, collection_path, category)
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error reading or processing JSON file: {str(e)}",
                "error": str(e)
            }

    def get_headlines_by_date(self, date: str) -> Dict[str, Any]:
        """
        Fetch all headlines for a specific date across all categories, excluding the "general" category.
        
        Args:
            date (str): Date in YYYY-MM-DD format
            
        Returns:
            Dict[str, Any]: Dictionary containing headlines organized by category
        """
        try:
            # Reference to the categories collection
            categories_ref = (
                self.db.collection("news_latest")
                .document("top_headlines")
                .collection("categories")
            )
            
            # Get all categories using list_documents
            categories = list(categories_ref.list_documents())
            
            result = {
                "success": True,
                "date": date,
                "categories": {}
            }
            
            # Iterate through each category
            for category_doc in categories:
                category = category_doc.id
                
                # Skip the "general" category
                if category == "general":
                    continue
                
                # Reference to the headlines collection for this category and date
                headlines_ref = (
                    categories_ref
                    .document(category)
                    .collection("dates")
                    .document(date)
                    .collection("headlines")
                )
                
                # Get all headlines for this category and date
                headlines = headlines_ref.get()
                
                # Store headlines for this category
                category_headlines = []
                for headline_doc in headlines:
                    headline_data = headline_doc.to_dict()
                    headline_data["id"] = headline_doc.id
                    category_headlines.append(headline_data)
                
                # Add category headlines to result if any exist
                if category_headlines:
                    result["categories"][category] = category_headlines
            
            # Add total headlines count
            total_headlines = sum(len(headlines) for headlines in result["categories"].values())
            result["total_headlines"] = total_headlines
            
            if total_headlines == 0:
                result["message"] = f"No headlines found for date: {date}"
            else:
                result["message"] = f"Successfully retrieved {total_headlines} headlines across {len(result['categories'])} categories"
            
            # Convert any Firestore timestamps to ISO format strings
            result = self._convert_timestamp_to_iso(result)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error fetching headlines: {str(e)}",
                "error": str(e)
            }

# Example usage:
if __name__ == "__main__":
    # Initialize the Firebase tools
    firebase_tools = FirebaseTools()
    
    # # Example of pushing headlines from a JSON file
    # json_path = "../../data/summary/top_headlines_summary.json"
    # result = firebase_tools.push_headlines_from_json(json_path, category="technology")
    # print(result)
    
    # Example of fetching headlines by date
    date = "2025-06-02"  # Replace with desired date
    headlines = firebase_tools.get_headlines_by_date(date)
    print(json.dumps(headlines, indent=2))