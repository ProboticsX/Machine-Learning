from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "newspresso"
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings()

def get_result(query, similar_result = 3):
  vector = embeddings.embed_query(query)
  result = index.query(vector = vector, top_k = similar_result)
  return result

query = "What happened with Taylor Swift in Paris?"
result = get_result(query)
print(result)