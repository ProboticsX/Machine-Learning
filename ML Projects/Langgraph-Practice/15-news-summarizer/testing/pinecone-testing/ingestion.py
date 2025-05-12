import json
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

class NewsVectorStore:
    def __init__(self):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # List existing indexes
        existing_indexes = pc.list_indexes()
        print(f"Existing indexes: {existing_indexes}")
        
        # Set the index name
        self.index_name = "news-articles"
        
        # Check if the index exists
        index_exists = any(index["name"] == self.index_name for index in existing_indexes)
        
        if not index_exists:
            print(f"Creating new index: {self.index_name}")
            try:
                # Create the index
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Successfully created new index: {self.index_name}")
            except Exception as e:
                print(f"Error creating index: {e}")
                raise
        else:
            print(f"Using existing index: {self.index_name}")
            
        self.index = pc.Index(self.index_name)

    def read_news_content(self, file_path: str) -> List[Dict]:
        """Read and parse the processed news JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data["articles"]

    def prepare_vectors(self, articles: List[Dict]) -> List[Dict]:
        """Prepare vectors for Pinecone insertion."""
        vectors = []
        for i, article in enumerate(articles):
            # Create a text representation of the article
            text = f"Title: {article['title']}\n"
            text += f"Description: {article['description']}\n"
            text += f"Content: {article['content']}\n"
            
            # Generate embedding
            embedding = self.embeddings.embed_query(text)
            
            # Create metadata
            metadata = {
                "source": article["source"],
                "title": article["title"],
                "description": article["description"],
                "url": article["url"],
                "published_at": article["published_at"],
                "author": article["author"] if article["author"] else "Unknown"
            }
            
            vectors.append({
                "id": f"article_{i}",
                "values": embedding,
                "metadata": metadata
            })
        
        return vectors

    def store_in_pinecone(self, vectors: List[Dict]):
        """Store vectors in Pinecone."""
        # Upsert vectors in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Path to the processed news file
    news_file = project_root / "testing" / "news-api-testing" / "news" / "processed_news.json"
    
    # Initialize the vector store
    vector_store = NewsVectorStore()
    
    # Read news content
    articles = vector_store.read_news_content(str(news_file))
    print(f"Read {len(articles)} articles from {news_file}")
    
    # Prepare vectors
    vectors = vector_store.prepare_vectors(articles)
    print(f"Prepared {len(vectors)} vectors for storage")
    
    # Store in Pinecone
    vector_store.store_in_pinecone(vectors)
    print("Successfully stored vectors in Pinecone")

if __name__ == "__main__":
    main()