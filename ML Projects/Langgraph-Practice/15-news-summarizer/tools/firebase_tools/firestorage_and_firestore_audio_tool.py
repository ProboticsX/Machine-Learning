from typing import List, Dict, Any, Optional
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
import json
import os
from pathlib import Path

class FireStorageAndFireStoreAudioTool:
    def __init__(self, service_account_path: str = None, storage_bucket: str = None):
        """
        Initialize Firebase connection.
        
        Args:
            service_account_path (str, optional): Path to Firebase service account JSON file.
                If not provided, will look for it in the current directory.
            storage_bucket (str, optional): Firebase Storage bucket name.
                If not provided, will use the default bucket for the project.
        """
        if not service_account_path:
            # Try to find the service account file in the current directory
            current_dir = Path(__file__).parent
            service_account_path = str(current_dir / "firebaseServiceAccountKey.json")
        
        if not os.path.exists(service_account_path):
            raise FileNotFoundError(f"Firebase service account file not found at: {service_account_path}")
        
        # Initialize Firebase if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            
            # Get the project ID from the service account file
            with open(service_account_path, 'r') as f:
                service_account = json.load(f)
                project_id = service_account.get('project_id')
            
            # Use provided storage bucket or construct default one
            if not storage_bucket:
                storage_bucket = f"{project_id}.firebasestorage.app"
            
            try:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': storage_bucket
                })
            except Exception as e:
                raise Exception(f"Failed to initialize Firebase: {str(e)}. Please ensure Firebase Storage is enabled for your project and the bucket name is correct.")
        
        self.db = firestore.client()
        try:
            # Explicitly specify the bucket name when getting the bucket
            self.bucket = storage.bucket(storage_bucket)
        except Exception as e:
            raise Exception(f"Failed to access Firebase Storage bucket: {str(e)}. Please ensure Firebase Storage is enabled for your project.")
    
    def _delete_existing_audio(self, today_date: str, category: str = "general") -> None:
        """
        Delete existing audio files for a given date from both Firestore and Storage.
        
        Args:
            today_date (str): Date in YYYY-MM-DD format
            category (str): Category of the headlines (e.g., technology, business, general)
        """
        try:
            # Reference to today's audio files collection
            # Structure: news_latest (collection) -> top_headlines (document) -> categories (collection) -> category (document) -> dates (collection) -> date (document) -> audio_files (collection)
            audio_ref = (
                self.db.collection("news_latest")
                .document("top_headlines")
                .collection("categories")
                .document(category)
                .collection("dates")
                .document(today_date)
                .collection("audio_files")
            )
            
            # Get all audio files for today
            existing_files = audio_ref.get()
            
            # Delete from Firestore and Storage
            for doc in existing_files:
                # Get the storage path from the document
                storage_path = doc.get("storage_path")
                if storage_path:
                    # Delete from Storage
                    blob = self.bucket.blob(storage_path)
                    if blob.exists():
                        blob.delete()
                
                # Delete from Firestore
                doc.reference.delete()
            
            print(f"Deleted existing audio files for {category} on {today_date}")
            
        except Exception as e:
            print(f"Error deleting existing audio files: {str(e)}")
    
    def store_audio_file(self, audio_file_path: str, category: str = "general", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store an audio file in Firebase Storage and save its metadata to Firestore.
        
        Args:
            audio_file_path (str): Path to the audio file
            category (str): Category of the headlines (e.g., technology, business, general)
            metadata (Dict[str, Any], optional): Additional metadata to store with the file
            
        Returns:
            Dict[str, Any]: Status of the operation including success/failure and details
        """
        try:
            if not os.path.exists(audio_file_path):
                return {
                    "success": False,
                    "message": f"Audio file not found at: {audio_file_path}",
                    "error": "FileNotFoundError"
                }
            
            # Get file information
            file_name = os.path.basename(audio_file_path)
            file_extension = os.path.splitext(file_name)[1].lower()
            
            # Validate file type
            if file_extension not in ['.mp3', '.wav', '.m4a', '.ogg']:
                return {
                    "success": False,
                    "message": f"Unsupported audio file format: {file_extension}",
                    "error": "UnsupportedFileFormat"
                }
            
            # Get today's date in YYYY-MM-DD format
            today_date = datetime.now().strftime("%Y-%m-%d")
            
            # Delete existing audio files for today
            self._delete_existing_audio(today_date, category)
            
            # Create a unique storage path
            storage_path = f"audio/{category}/{today_date}/{file_name}"
            
            try:
                # Upload file to Firebase Storage
                blob = self.bucket.blob(storage_path)
                blob.upload_from_filename(audio_file_path)
                
                # Make the file publicly accessible
                blob.make_public()
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to upload file to Firebase Storage: {str(e)}. Please ensure Firebase Storage is properly configured.",
                    "error": str(e)
                }
            
            # Prepare minimal metadata for Firestore
            file_metadata = {
                "file_name": file_name,
                "storage_path": storage_path,
                "public_url": blob.public_url,
                "category": category,
                "date": today_date,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            
            # Save metadata to Firestore using the same structure as headlines
            # Structure: news_latest (collection) -> top_headlines (document) -> categories (collection) -> category (document) -> dates (collection) -> date (document) -> audio_files (collection)
            metadata_ref = (
                self.db.collection("news_latest")
                .document("top_headlines")
                .collection("categories")
                .document(category)
                .collection("dates")
                .document(today_date)
                .collection("audio_files")
                .document(file_name)
            )
            metadata_ref.set(file_metadata)
            
            return {
                "success": True,
                "message": f"Successfully uploaded audio file: {file_name}",
                "file_name": file_name,
                "public_url": blob.public_url,
                "storage_path": storage_path,
                "category": category,
                "date": today_date
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error storing audio file: {str(e)}",
                "error": str(e)
            }
    
    def store_audio_directory(self, directory_path: str, category: str = "general") -> Dict[str, Any]:
        """
        Store all audio files from a directory in Firebase Storage.
        
        Args:
            directory_path (str): Path to the directory containing audio files
            category (str): Category of the headlines (e.g., technology, business, general)
            
        Returns:
            Dict[str, Any]: Status of the operation including success/failure and details
        """
        try:
            if not os.path.exists(directory_path):
                return {
                    "success": False,
                    "message": f"Directory not found at: {directory_path}",
                    "error": "DirectoryNotFoundError"
                }
            
            # Get today's date
            today_date = datetime.now().strftime("%Y-%m-%d")
            
            # Delete existing audio files for today
            self._delete_existing_audio(today_date, category)
            
            results = []
            for file_name in os.listdir(directory_path):
                if file_name.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg')):
                    file_path = os.path.join(directory_path, file_name)
                    result = self.store_audio_file(file_path, category)
                    results.append(result)
            
            return {
                "success": True,
                "message": f"Processed {len(results)} audio files",
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing audio directory: {str(e)}",
                "error": str(e)
            }

    # ... (keep existing push_headlines and push_headlines_from_json methods) ...

# Example usage:
if __name__ == "__main__":
    # Initialize the Firebase tools with the correct bucket name
    firebase_tools = FireStorageAndFireStoreAudioTool(storage_bucket="iosapp-5d233.firebasestorage.app")
    
    # Example of storing an audio file
    audio_path = "../../data/audio/podcast_365256d0447a49ac912115d52cbaa0e1.mp3"
    result = firebase_tools.store_audio_file(audio_path, category="technology")
    print(result)
    
    # Example of storing all audio files in a directory
    # audio_dir = "../../data/audio"
    # result = firebase_tools.store_audio_directory(audio_dir, category="technology")
    # print(result) 