import json
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

class HeadlineIngestor:
    def __init__(self):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Initialize TF-IDF vectorizer for sparse vectors
        self.tfidf = TfidfVectorizer(max_features=1000)
        
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

    def _create_sparse_vector(self, text: str) -> Dict[str, List]:
        """Create a sparse vector using TF-IDF.
        
        Returns:
            Dict[str, List]: A dictionary with 'indices' and 'values' lists for sparse vector format
        """
        # Fit and transform the text
        tfidf_matrix = self.tfidf.fit_transform([text])
        
        # Get feature names and values
        feature_names = self.tfidf.get_feature_names_out()
        values = tfidf_matrix.toarray()[0]
        
        # Create sparse vector in Pinecone format
        indices = []
        vector_values = []
        
        for i, value in enumerate(values):
            if value > 0:  # Only include non-zero values
                indices.append(i)
                vector_values.append(float(value))
        
        return {
            "indices": indices,
            "values": vector_values
        }

    def ingest_headlines(self, 
                        json_file_path: str, 
                        index_name: str = "newspresso",
                        date: Optional[str] = None,
                        category: Optional[str] = None):
        """Ingest headlines from JSON file into Pinecone.
        
        Args:
            json_file_path (str): Path to the JSON file containing headlines
            index_name (str): Base name for the Pinecone indexes (default: "newspresso")
            date (str, optional): Override date for all headlines (format: YYYY-MM-DD)
            category (str, optional): Category to assign to all headlines
        """
        # Check if the indexes exist
        existing_indexes = self.pc.list_indexes()
        dense_index_name = f"{index_name}-dense"
        sparse_index_name = f"{index_name}-sparse"
        
        dense_index_exists = any(index["name"] == dense_index_name for index in existing_indexes)
        sparse_index_exists = any(index["name"] == sparse_index_name for index in existing_indexes)
        
        # Create dense index if it doesn't exist
        if not dense_index_exists:
            print(f"Creating new dense index: {dense_index_name}")
            try:
                self.pc.create_index(
                    name=dense_index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Successfully created new dense index: {dense_index_name}")
            except Exception as e:
                print(f"Error creating dense index: {e}")
                raise
        else:
            print(f"Using existing dense index: {dense_index_name}")
            
        # Create sparse index if it doesn't exist
        if not sparse_index_exists:
            print(f"Creating new sparse index: {sparse_index_name}")
            try:
                self.pc.create_index(
                    name=sparse_index_name,
                    vector_type="sparse",
                    metric="dotproduct",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Successfully created new sparse index: {sparse_index_name}")
            except Exception as e:
                print(f"Error creating sparse index: {e}")
                raise
        else:
            print(f"Using existing sparse index: {sparse_index_name}")
            
        self.dense_index = self.pc.Index(dense_index_name)
        self.sparse_index = self.pc.Index(sparse_index_name)
        
        # Read JSON file
        with open(json_file_path, 'r') as f:
            headlines = json.load(f)
        
        # Process headlines
        dense_vectors = []
        sparse_vectors = []
        
        # First pass: collect all texts for TF-IDF fitting
        all_texts = []
        for headline in headlines:
            headline_data = self._prepare_headline_data(headline, category)
            all_texts.append(headline_data["text"])
        
        # Fit TF-IDF on all texts
        self.tfidf.fit(all_texts)
        
        # Second pass: create vectors
        for i, headline in enumerate(headlines):
            # Prepare headline data
            headline_data = self._prepare_headline_data(headline, category)
            
            # Generate dense embedding
            dense_embedding = self.embeddings.embed_query(headline_data["text"])
            
            # Generate sparse vector
            sparse_vector = self._create_sparse_vector(headline_data["text"])
            
            # Create composite ID
            headline_date = date or headline.get("published_at", "unknown_date")
            if category:
                composite_id = f"{headline_date}_{category}_headline_{i}"
            else:
                composite_id = f"{headline_date}_headline_{i}"
            
            # Prepare dense vector
            dense_vectors.append({
                "id": composite_id,
                "values": dense_embedding,
                "metadata": headline_data["metadata"]
            })
            
            # Prepare sparse vector
            sparse_vectors.append({
                "id": composite_id,
                "sparse_values": {  # Only include sparse_values for sparse indexes
                    "indices": sparse_vector["indices"],
                    "values": sparse_vector["values"]
                },
                "metadata": headline_data["metadata"]
            })
        
        # Store vectors in batches of 100
        batch_size = 100
        for i in range(0, len(dense_vectors), batch_size):
            dense_batch = dense_vectors[i:i + batch_size]
            sparse_batch = sparse_vectors[i:i + batch_size]
            
            # Upsert to both indexes
            self.dense_index.upsert(vectors=dense_batch)
            self.sparse_index.upsert(vectors=sparse_batch)
        
        print(f"Successfully ingested {len(headlines)} headlines into:")
        print(f"- Dense index: '{dense_index_name}'")
        print(f"- Sparse index: '{sparse_index_name}'")
        if date:
            print(f"Using provided date: {date}")
        if category:
            print(f"Using provided category: {category}")

class HeadlineRetriever:
    def __init__(self, index_name: str = "newspresso"):
        """Initialize the retriever with Pinecone indexes and OpenAI embeddings.
        
        Args:
            index_name (str): Base name for the Pinecone indexes to use (default: "newspresso")
        """
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.dense_index = self.pc.Index(f"{index_name}-dense")
        self.sparse_index = self.pc.Index(f"{index_name}-sparse")
        
        # Initialize TF-IDF vectorizer for sparse vectors
        self.tfidf = TfidfVectorizer(max_features=1000)
    
    def _create_sparse_vector(self, text: str) -> Dict[str, List]:
        """Create a sparse vector using TF-IDF."""
        # Fit and transform the text
        tfidf_matrix = self.tfidf.fit_transform([text])
        
        # Get feature names and values
        feature_names = self.tfidf.get_feature_names_out()
        values = tfidf_matrix.toarray()[0]
        
        # Create sparse vector in Pinecone format
        indices = []
        vector_values = []
        
        for i, value in enumerate(values):
            if value > 0:  # Only include non-zero values
                indices.append(i)
                vector_values.append(float(value))
        
        return {
            "indices": indices,
            "values": vector_values
        }
    
    def _merge_and_deduplicate_results(self, dense_results, sparse_results, top_k: int) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from dense and sparse searches."""
        # Create a dictionary to store unique results by ID
        unique_results = {}
        
        # Process dense results
        for match in dense_results.matches:
            unique_results[match.id] = {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
                "date": match.id.split('_')[0]
            }
        
        # Process sparse results and update scores if ID exists
        for match in sparse_results.matches:
            if match.id in unique_results:
                # If ID exists, take the higher score
                unique_results[match.id]["score"] = max(unique_results[match.id]["score"], match.score)
            else:
                unique_results[match.id] = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "date": match.id.split('_')[0]
                }
        
        # Convert to list and sort by score
        merged_results = list(unique_results.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        return merged_results[:top_k]
    
    def retrieve_headlines(self, 
                          query: str, 
                          top_k: int = 3,
                          date: Optional[str] = None,
                          alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Retrieve relevant headlines using hybrid search.
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return (default: 3)
            date (str, optional): Filter results by date (format: YYYY-MM-DD)
            alpha (float): Weight for dense vs sparse search (0-1, default: 0.5)
            
        Returns:
            List[Dict[str, Any]]: List of retrieved headlines with their metadata
        """
        # Generate dense embedding for the query
        dense_embedding = self.embeddings.embed_query(query)
        
        # Generate sparse vector for the query
        sparse_vector = self._create_sparse_vector(query)
        
        # Query both indexes
        dense_results = self.dense_index.query(
            vector=dense_embedding,
            top_k=top_k * 2,  # Fetch more results to account for filtering
            include_metadata=True
        )
        
        sparse_results = self.sparse_index.query(
            sparse_vector=sparse_vector,
            top_k=top_k * 2,  # Fetch more results to account for filtering
            include_metadata=True
        )
        
        # Merge and deduplicate results
        merged_results = self._merge_and_deduplicate_results(dense_results, sparse_results, top_k * 2)
        
        # Apply date filter if specified
        if date:
            merged_results = [r for r in merged_results if r["date"] == date]
        
        # Return top_k results
        return merged_results[:top_k]

def main():
    try:
        # Initialize retriever with existing indices
        retriever = HeadlineRetriever(index_name="newspresso")
        
        # Test queries based on the actual headlines
        test_queries = [
            "How did Trump's meeting with the German Chancellor go?"
        ]
        
        print("\nTesting hybrid search retrieval:")
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.retrieve_headlines(query, top_k=2)
            print(results)
            for result in results:
                print(f"\nTitle: {result['metadata']['title']}")
                print(f"Date: {result['date']}")
                print(f"Score: {result['score']:.4f}")
                print(f"Summary: {result['metadata']['content_summary']}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
