import json
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class HeadlineIngestor:
    def __init__(self):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # List existing indexes
        existing_indexes = self.pc.list_indexes()
        print(f"Existing indexes: {existing_indexes}")

    def _prepare_headline_data(self, headline: Dict[str, Any], category: Optional[str] = None) -> Dict[str, Any]:
        """Prepare headline data for vector storage."""
        # Create a rich text representation for embedding
        text = f"""
        Title: {headline['title']}
        Description: {headline['description']}
        Content Summary: {headline['content_summary']}
        Published At: {headline.get('published_at', 'N/A')}
        """
        
        if category:
            text += f"Category: {category}\n"
        
        # Create metadata
        metadata = {
            "title": headline["title"],
            "description": headline["description"],
            "content_summary": headline["content_summary"],
            "published_at": headline.get("published_at", "N/A"),
            "sources": headline.get("sources", []),
            "urlToImage": headline.get("urlToImage", "")
        }
        
        if category:
            metadata["category"] = category
        
        return {"text": text, "metadata": metadata}

    def ingest_headlines(self, 
                        json_file_path: str, 
                        index_name: str = "newspresso",
                        date: Optional[str] = None,
                        category: Optional[str] = None):
        """Ingest headlines from JSON file into Pinecone.
        
        Args:
            json_file_path (str): Path to the JSON file containing headlines
            index_name (str): Name of the Pinecone index to use (default: "newspresso")
            date (str, optional): Override date for all headlines (format: YYYY-MM-DD)
            category (str, optional): Category to assign to all headlines
        """
        # Check if the index exists
        existing_indexes = self.pc.list_indexes()
        index_exists = any(index["name"] == index_name for index in existing_indexes)
        
        if not index_exists:
            print(f"Creating new index: {index_name}")
            try:
                # Create the index
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Successfully created new index: {index_name}")
            except Exception as e:
                print(f"Error creating index: {e}")
                raise
        else:
            print(f"Using existing index: {index_name}")
            
        self.index = self.pc.Index(index_name)
        
        # Read JSON file
        with open(json_file_path, 'r') as f:
            headlines = json.load(f)
        
        # Process headlines
        vectors = []
        for i, headline in enumerate(headlines):
            # Prepare headline data
            headline_data = self._prepare_headline_data(headline, category)
            
            # Generate embedding
            embedding = self.embeddings.embed_query(headline_data["text"])
            
            # Create composite ID: date_headline[index] or date_category_headline[index]
            headline_date = date or headline.get("published_at", "unknown_date")
            if category:
                composite_id = f"{headline_date}_{category}_headline_{i}"
            else:
                composite_id = f"{headline_date}_headline_{i}"
            
            # Prepare vector
            vectors.append({
                "id": composite_id,
                "values": embedding,
                "metadata": headline_data["metadata"]
            })
        
        # Store vectors in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Successfully ingested {len(headlines)} headlines into Pinecone index '{index_name}'")
        if date:
            print(f"Using provided date: {date}")
        if category:
            print(f"Using provided category: {category}")

def main():
    # Initialize ingestor
    ingestor = HeadlineIngestor()
    
    # Ingest headlines
    json_file_path = "./top_headlines_summary.json"
    index_name = "newspresso"
    date = "2025-05-31"
    category = "business"
    
    ingestor.ingest_headlines(
        json_file_path=json_file_path,
        index_name=index_name,
        date=date,
        category=category
    )

if __name__ == "__main__":
    main()
