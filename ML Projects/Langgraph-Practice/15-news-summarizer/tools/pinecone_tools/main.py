import json
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Characters per chunk
            chunk_overlap=200,  # Overlap between chunks
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # List existing indexes
        existing_indexes = self.pc.list_indexes()
        print(f"Existing indexes: {existing_indexes}")

    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks and prepare them for ingestion.
        
        Args:
            text (str): The text to chunk
            metadata (Dict[str, Any]): Original metadata to attach to each chunk
            
        Returns:
            List[Dict[str, Any]]: List of chunked texts with minimal metadata
        """
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Prepare chunks with minimal metadata
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            # Create minimal metadata with only required fields
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),  # Added back total_chunks
                "chunk_text": chunk,
                "published_at": metadata.get("published_at", "N/A"),
                "category": metadata.get("category", "general")
            }
            
            chunked_docs.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        return chunked_docs

    def _prepare_headline_data(self, headline: Dict[str, Any], category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Prepare headline data for vector storage with chunking.
        
        Returns:
            List[Dict[str, Any]]: List of chunked documents ready for ingestion
        """
        # Create a rich text representation for embedding
        text = f"""
        Title: {headline['title']}
        Description: {headline['description']}
        Content Summary: {headline['content_summary']}
        """
        
        # Create minimal base metadata with only required fields
        metadata = {
            "published_at": headline.get("published_at", "N/A"),
            "category": category if category else "uncategorized"
        }
        
        # Chunk the text and prepare documents
        return self._chunk_text(text, metadata)

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
        """Ingest headlines from JSON file into Pinecone with chunking."""
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
            chunked_docs = self._prepare_headline_data(headline, category)
            for doc in chunked_docs:
                all_texts.append(doc["text"])
        
        # Fit TF-IDF on all texts
        self.tfidf.fit(all_texts)
        
        # Second pass: create vectors
        for i, headline in enumerate(headlines):
            # Prepare headline data with chunking
            chunked_docs = self._prepare_headline_data(headline, category)
            
            for chunk_idx, doc in enumerate(chunked_docs):
                # Generate dense embedding
                dense_embedding = self.embeddings.embed_documents([doc["text"]])[0]
                
                # Generate sparse vector
                sparse_vector = self._create_sparse_vector(doc["text"])
                
                # Create composite ID that includes chunk information
                headline_date = date or headline.get("published_at", "unknown_date")
                if category:
                    composite_id = f"{headline_date}_{category}_headline_{i}_chunk_{chunk_idx}"
                else:
                    composite_id = f"{headline_date}_headline_{i}_chunk_{chunk_idx}"
                
                # Prepare dense vector with minimal metadata
                dense_vectors.append({
                    "id": composite_id,
                    "values": dense_embedding,
                    "metadata": doc["metadata"]  # Now contains only category, chunk_index, chunk_text, published_at
                })
                
                # Prepare sparse vector with minimal metadata
                sparse_vectors.append({
                    "id": composite_id,
                    "sparse_values": {
                        "indices": sparse_vector["indices"],
                        "values": sparse_vector["values"]
                    },
                    "metadata": doc["metadata"]  # Now contains only category, chunk_index, chunk_text, published_at
                })
        
        # Store vectors in batches of 100
        batch_size = 100
        for i in range(0, len(dense_vectors), batch_size):
            dense_batch = dense_vectors[i:i + batch_size]
            sparse_batch = sparse_vectors[i:i + batch_size]
            
            # Upsert to both indexes
            self.dense_index.upsert(vectors=dense_batch)
            self.sparse_index.upsert(vectors=sparse_batch)
        
        total_chunks = len(dense_vectors)
        print(f"Successfully ingested {len(headlines)} headlines ({total_chunks} chunks) into:")
        print(f"- Dense index: '{dense_index_name}'")
        print(f"- Sparse index: '{sparse_index_name}'")
        print(f"Metadata fields stored: category, chunk_index, total_chunks, chunk_text, published_at")
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
    
    def _merge_and_deduplicate_results(self, dense_results, sparse_results, top_k: int, query: str) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from dense and sparse searches."""
        # Create a dictionary to store unique results by ID
        unique_results = {}
        
        # Process dense results
        for match in dense_results.matches:
            unique_results[match.id] = {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
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
                    "metadata": match.metadata
                }
        
        # Convert to list and sort by score
        merged_results = list(unique_results.values())
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Prepare documents for reranking in the correct format
        rerank_docs = []
        for r in merged_results:
            # Get the text content from metadata (we know it's stored in chunk_text)
            text_content = r["metadata"].get("chunk_text", "")
            
            # Only add documents that have content
            if text_content:
                rerank_docs.append({
                    "text": text_content,
                    "id": r["id"]
                })
        
        if not rerank_docs:
            return merged_results[:top_k]
            
        try:
            # Call reranker with the correct document format
            rerank_result = self.pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query=query,
                documents=rerank_docs,
                top_n=min(top_k, len(rerank_docs)),
                return_documents=True
            )
            
            # Map reranked order back to full metadata
            id_to_result = {r["id"]: r for r in merged_results}
            reranked = []
            
            for row in rerank_result.data:
                doc_id = row["document"]["id"]
                if doc_id in id_to_result:
                    # Create result with only the metadata we store
                    result = {
                        "id": doc_id,
                        "score": id_to_result[doc_id]["score"],
                        "rerank_score": row["score"],
                        "metadata": {
                            "category": id_to_result[doc_id]["metadata"]["category"],
                            "chunk_index": id_to_result[doc_id]["metadata"]["chunk_index"],
                            "total_chunks": id_to_result[doc_id]["metadata"]["total_chunks"],
                            "chunk_text": id_to_result[doc_id]["metadata"]["chunk_text"],
                            "published_at": id_to_result[doc_id]["metadata"]["published_at"]
                        }
                    }
                    reranked.append(result)
            
            return reranked[:top_k]
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            # Fallback to original results if reranking fails
            return merged_results[:top_k]
    
    def retrieve_headlines(self, 
                          query: str, 
                          top_k: int = 5,
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
        merged_results = self._merge_and_deduplicate_results(dense_results, sparse_results, top_k * 2, query)
        
        # Apply date filter if specified
        if date:
            merged_results = [r for r in merged_results if r["metadata"]["published_at"] == date]
        
        # Return top_k results
        return merged_results[:top_k]

def main():
    try:
        # ingestor = HeadlineIngestor()
        # ingestor.ingest_headlines(json_file_path="./top_headlines_summary.json", index_name="newspresso", date="2025-06-12", category="business")

        # Initialize retriever with existing indices
        retriever = HeadlineRetriever(index_name="newspresso")
        
        # Test queries based on the actual headlines
        test_queries = [
            "Any news in entertainment?"
        ]
        
        print("\nTesting hybrid search retrieval:")
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.retrieve_headlines(query, top_k=5)
            for result in results:
                print(f"\nCategory: {result['metadata']['category']}")
                print(f"Published At: {result['metadata']['published_at']}")
                print(f"Chunk {result['metadata']['chunk_index'] + 1} of {result['metadata']['total_chunks']}")
                print(f"Score: {result['score']:.4f}")
                print(f"Rerank Score: {result['rerank_score']:.4f}")
                print(f"Content: {result['metadata']['chunk_text']}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

# def merge_chunks(h1, h2):
#     """Get the unique hits from two search results and return them as single array of {'_id', 'chunk_text'} dicts, printing each dict on a new line."""
#     # Deduplicate by _id
#     deduped_hits = {hit['_id']: hit for hit in h1['result']['hits'] + h2['result']['hits']}.values()
#     # Sort by _score descending
#     sorted_hits = sorted(deduped_hits, key=lambda x: x['_score'], reverse=True)
#     # Transform to format for reranking
#     result = [{'_id': hit['_id'], 'chunk_text': hit['fields']['chunk_text']} for hit in sorted_hits]
#     return result


if __name__ == "__main__":
    main()
    # pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # dense_index_name = "dense-for-hybrid-py"
    # sparse_index_name = "sparse-for-hybrid-py"

    # if not pc.has_index(dense_index_name):
    #     pc.create_index_for_model(
    #         name=dense_index_name,
    #         cloud="aws",
    #         region="us-east-1",
    #         embed={
    #             "model":"llama-text-embed-v2",
    #             "field_map":{"text": "chunk_text"}
    #         }
    #     )

    # if not pc.has_index(sparse_index_name):
    #     pc.create_index_for_model(
    #         name=sparse_index_name,
    #         cloud="aws",
    #         region="us-east-1",
    #         embed={
    #             "model":"pinecone-sparse-english-v0",
    #             "field_map":{"text": "chunk_text"}
    #         }
    #     )
    # # Define the records
    # records = [
    # { "_id": "vec1", "chunk_text": "Apple Inc. issued a $10 billion corporate bond in 2023." },
    # { "_id": "vec2", "chunk_text": "ETFs tracking the S&P 500 outperformed active funds last year." },
    # { "_id": "vec3", "chunk_text": "Tesla's options volume surged after the latest earnings report." },
    # { "_id": "vec4", "chunk_text": "Dividend aristocrats are known for consistently raising payouts." },
    # { "_id": "vec5", "chunk_text": "The Federal Reserve raised interest rates by 0.25% to curb inflation." },
    # { "_id": "vec6", "chunk_text": "Unemployment hit a record low of 3.7% in Q4 of 2024." },
    # { "_id": "vec7", "chunk_text": "The CPI index rose by 6% in July 2024, raising concerns about purchasing power." },
    # { "_id": "vec8", "chunk_text": "GDP growth in emerging markets outpaced developed economies." },
    # { "_id": "vec9", "chunk_text": "Amazon's acquisition of MGM Studios was valued at $8.45 billion." },
    # { "_id": "vec10", "chunk_text": "Alphabet reported a 20% increase in advertising revenue." },
    # { "_id": "vec11", "chunk_text": "ExxonMobil announced a special dividend after record profits." },
    # { "_id": "vec12", "chunk_text": "Tesla plans a 3-for-1 stock split to attract retail investors." },
    # { "_id": "vec13", "chunk_text": "Credit card APRs reached an all-time high of 22.8% in 2024." },
    # { "_id": "vec14", "chunk_text": "A 529 college savings plan offers tax advantages for education." },
    # { "_id": "vec15", "chunk_text": "Emergency savings should ideally cover 6 months of expenses." },
    # { "_id": "vec16", "chunk_text": "The average mortgage rate rose to 7.1% in December." },
    # { "_id": "vec17", "chunk_text": "The SEC fined a hedge fund $50 million for insider trading." },
    # { "_id": "vec18", "chunk_text": "New ESG regulations require companies to disclose climate risks." },
    # { "_id": "vec19", "chunk_text": "The IRS introduced a new tax bracket for high earners." },
    # { "_id": "vec20", "chunk_text": "Compliance with GDPR is mandatory for companies operating in Europe." },
    # { "_id": "vec21", "chunk_text": "What are the best-performing green bonds in a rising rate environment?" },
    # { "_id": "vec22", "chunk_text": "How does inflation impact the real yield of Treasury bonds?" },
    # { "_id": "vec23", "chunk_text": "Top SPAC mergers in the technology sector for 2024." },
    # { "_id": "vec24", "chunk_text": "Are stablecoins a viable hedge against currency devaluation?" },
    # { "_id": "vec25", "chunk_text": "Comparison of Roth IRA vs 401(k) for high-income earners." },
    # { "_id": "vec26", "chunk_text": "Stock splits and their effect on investor sentiment." },
    # { "_id": "vec27", "chunk_text": "Tech IPOs that disappointed in their first year." },
    # { "_id": "vec28", "chunk_text": "Impact of interest rate hikes on bank stocks." },
    # { "_id": "vec29", "chunk_text": "Growth vs. value investing strategies in 2024." },
    # { "_id": "vec30", "chunk_text": "The role of artificial intelligence in quantitative trading." },
    # { "_id": "vec31", "chunk_text": "What are the implications of quantitative tightening on equities?" },
    # { "_id": "vec32", "chunk_text": "How does compounding interest affect long-term investments?" },
    # { "_id": "vec33", "chunk_text": "What are the best assets to hedge against inflation?" },
    # { "_id": "vec34", "chunk_text": "Can ETFs provide better diversification than mutual funds?" },
    # { "_id": "vec35", "chunk_text": "Unemployment hit at 2.4% in Q3 of 2024." },
    # { "_id": "vec36", "chunk_text": "Unemployment is expected to hit 2.5% in Q3 of 2024." },
    # { "_id": "vec37", "chunk_text": "In Q3 2025 unemployment for the prior year was revised to 2.2%"},
    # { "_id": "vec38", "chunk_text": "Emerging markets witnessed increased foreign direct investment as global interest rates stabilized." },
    # { "_id": "vec39", "chunk_text": "The rise in energy prices significantly impacted inflation trends during the first half of 2024." },
    # { "_id": "vec40", "chunk_text": "Labor market trends show a declining participation rate despite record low unemployment in 2024." },
    # { "_id": "vec41", "chunk_text": "Forecasts of global supply chain disruptions eased in late 2024, but consumer prices remained elevated due to persistent demand." },
    # { "_id": "vec42", "chunk_text": "Tech sector layoffs in Q3 2024 have reshaped hiring trends across high-growth industries." },
    # { "_id": "vec43", "chunk_text": "The U.S. dollar weakened against a basket of currencies as the global economy adjusted to shifting trade balances." },
    # { "_id": "vec44", "chunk_text": "Central banks worldwide increased gold reserves to hedge against geopolitical and economic instability." },
    # { "_id": "vec45", "chunk_text": "Corporate earnings in Q4 2024 were largely impacted by rising raw material costs and currency fluctuations." },
    # { "_id": "vec46", "chunk_text": "Economic recovery in Q2 2024 relied heavily on government spending in infrastructure and green energy projects." },
    # { "_id": "vec47", "chunk_text": "The housing market saw a rebound in late 2024, driven by falling mortgage rates and pent-up demand." },
    # { "_id": "vec48", "chunk_text": "Wage growth outpaced inflation for the first time in years, signaling improved purchasing power in 2024." },
    # { "_id": "vec49", "chunk_text": "China's economic growth in 2024 slowed to its lowest level in decades due to structural reforms and weak exports." },
    # { "_id": "vec50", "chunk_text": "AI-driven automation in the manufacturing sector boosted productivity but raised concerns about job displacement." },
    # { "_id": "vec51", "chunk_text": "The European Union introduced new fiscal policies in 2024 aimed at reducing public debt without stifling growth." },
    # { "_id": "vec52", "chunk_text": "Record-breaking weather events in early 2024 have highlighted the growing economic impact of climate change." },
    # { "_id": "vec53", "chunk_text": "Cryptocurrencies faced regulatory scrutiny in 2024, leading to volatility and reduced market capitalization." },
    # { "_id": "vec54", "chunk_text": "The global tourism sector showed signs of recovery in late 2024 after years of pandemic-related setbacks." },
    # { "_id": "vec55", "chunk_text": "Trade tensions between the U.S. and China escalated in 2024, impacting global supply chains and investment flows." },
    # { "_id": "vec56", "chunk_text": "Consumer confidence indices remained resilient in Q2 2024 despite fears of an impending recession." },
    # { "_id": "vec57", "chunk_text": "Startups in 2024 faced tighter funding conditions as venture capitalists focused on profitability over growth." },
    # { "_id": "vec58", "chunk_text": "Oil production cuts in Q1 2024 by OPEC nations drove prices higher, influencing global energy policies." },
    # { "_id": "vec59", "chunk_text": "The adoption of digital currencies by central banks increased in 2024, reshaping monetary policy frameworks." },
    # { "_id": "vec60", "chunk_text": "Healthcare spending in 2024 surged as governments expanded access to preventive care and pandemic preparedness." },
    # { "_id": "vec61", "chunk_text": "The World Bank reported declining poverty rates globally, but regional disparities persisted." },
    # { "_id": "vec62", "chunk_text": "Private equity activity in 2024 focused on renewable energy and technology sectors amid shifting investor priorities." },
    # { "_id": "vec63", "chunk_text": "Population aging emerged as a critical economic issue in 2024, especially in advanced economies." },
    # { "_id": "vec64", "chunk_text": "Rising commodity prices in 2024 strained emerging markets dependent on imports of raw materials." },
    # { "_id": "vec65", "chunk_text": "The global shipping industry experienced declining freight rates in 2024 due to overcapacity and reduced demand." },
    # { "_id": "vec66", "chunk_text": "Bank lending to small and medium-sized enterprises surged in 2024 as governments incentivized entrepreneurship." },
    # { "_id": "vec67", "chunk_text": "Renewable energy projects accounted for a record share of global infrastructure investment in 2024." },
    # { "_id": "vec68", "chunk_text": "Cybersecurity spending reached new highs in 2024, reflecting the growing threat of digital attacks on infrastructure." },
    # { "_id": "vec69", "chunk_text": "The agricultural sector faced challenges in 2024 due to extreme weather and rising input costs." },
    # { "_id": "vec70", "chunk_text": "Consumer spending patterns shifted in 2024, with a greater focus on experiences over goods." },
    # { "_id": "vec71", "chunk_text": "The economic impact of the 2008 financial crisis was mitigated by quantitative easing policies." },
    # { "_id": "vec72", "chunk_text": "In early 2024, global GDP growth slowed, driven by weaker exports in Asia and Europe." },
    # { "_id": "vec73", "chunk_text": "The historical relationship between inflation and unemployment is explained by the Phillips Curve." },
    # { "_id": "vec74", "chunk_text": "The World Trade Organization's role in resolving disputes was tested in 2024." },
    # { "_id": "vec75", "chunk_text": "The collapse of Silicon Valley Bank raised questions about regulatory oversight in 2024." },
    # { "_id": "vec76", "chunk_text": "The cost of living crisis has been exacerbated by stagnant wage growth and rising inflation." },
    # { "_id": "vec77", "chunk_text": "Supply chain resilience became a top priority for multinational corporations in 2024." },
    # { "_id": "vec78", "chunk_text": "Consumer sentiment surveys in 2024 reflected optimism despite high interest rates." },
    # { "_id": "vec79", "chunk_text": "The resurgence of industrial policy in Q1 2024 focused on decoupling critical supply chains." },
    # { "_id": "vec80", "chunk_text": "Technological innovation in the fintech sector disrupted traditional banking in 2024." },
    # { "_id": "vec81", "chunk_text": "The link between climate change and migration patterns is increasingly recognized." },
    # { "_id": "vec82", "chunk_text": "Renewable energy subsidies in 2024 reduced the global reliance on fossil fuels." },
    # { "_id": "vec83", "chunk_text": "The economic fallout of geopolitical tensions was evident in rising defense budgets worldwide." },
    # { "_id": "vec84", "chunk_text": "The IMF's 2024 global outlook highlighted risks of stagflation in emerging markets." },
    # { "_id": "vec85", "chunk_text": "Declining birth rates in advanced economies pose long-term challenges for labor markets." },
    # { "_id": "vec86", "chunk_text": "Digital transformation initiatives in 2024 drove productivity gains in the services sector." },
    # { "_id": "vec87", "chunk_text": "The U.S. labor market's resilience in 2024 defied predictions of a severe recession." },
    # { "_id": "vec88", "chunk_text": "New fiscal measures in the European Union aimed to stabilize debt levels post-pandemic." },
    # { "_id": "vec89", "chunk_text": "Venture capital investments in 2024 leaned heavily toward AI and automation startups." },
    # { "_id": "vec90", "chunk_text": "The surge in e-commerce in 2024 was facilitated by advancements in logistics technology." },
    # { "_id": "vec91", "chunk_text": "The impact of ESG investing on corporate strategies has been a major focus in 2024." },
    # { "_id": "vec92", "chunk_text": "Income inequality widened in 2024 despite strong economic growth in developed nations." },
    # { "_id": "vec93", "chunk_text": "The collapse of FTX highlighted the volatility and risks associated with cryptocurrencies." },
    # { "_id": "vec94", "chunk_text": "Cyberattacks targeting financial institutions in 2024 led to record cybersecurity spending." },
    # { "_id": "vec95", "chunk_text": "Automation in agriculture in 2024 increased yields but displaced rural workers." },
    # { "_id": "vec96", "chunk_text": "New trade agreements signed 2022 will make an impact in 2024"},
    # ]
    # dense_index = pc.Index(dense_index_name)
    # sparse_index = pc.Index(sparse_index_name)

    # # Upsert the records
    # # The `chunk_text` fields are converted to dense and sparse vectors
    # dense_index.upsert_records("example-namespace", records)
    # sparse_index.upsert_records("example-namespace", records)

    # query = "Does it mention anything related to China?"

    # dense_results = dense_index.search(
    #     namespace="example-namespace",
    #     query={
    #         "top_k": 40,
    #         "inputs": {
    #             "text": query
    #         }
    #     }
    # )

    # print(f"Dense results (Semantic Search):{dense_results}")

    # sparse_results = sparse_index.search(
    # namespace="example-namespace",
    # query={
    #         "top_k": 40,
    #         "inputs": {
    #             "text": query
    #         }
    #     }
    # )

    # print(f"Sparse Results (Lexical Search):{sparse_results}")

    # print("Merged Results:")
    # merged_results = merge_chunks(sparse_results, dense_results)

    # print('[\n   ' + ',\n   '.join(str(obj) for obj in merged_results) + '\n]')
    # result = pc.inference.rerank(
    # model="bge-reranker-v2-m3",
    # query=query,
    # documents=merged_results,
    # rank_fields=["chunk_text"],
    # top_n=40,
    # return_documents=True,
    # parameters={
    #     "truncate": "END"
    #     }
    # )

    # print("Query", query)
    # print('-----')
    # for row in result.data:
    #     print(f"{row['document']['_id']} - {round(row['score'], 2)} - {row['document']['chunk_text']}")