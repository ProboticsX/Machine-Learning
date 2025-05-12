from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "news-articles"
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings()

def get_result(query, similar_result = 3):
  vector = embeddings.embed_query(query)
  result = index.query(vector = vector, top_k = similar_result)
  return result

query = "How's the relation between India and Pakistan?"
result = get_result(query)
print(result)