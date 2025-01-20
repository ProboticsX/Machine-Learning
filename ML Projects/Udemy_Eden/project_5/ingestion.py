import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import embeddings

load_dotenv()
embeddings = OpenAIEmbeddings()

def ingest_docs():
    loader = ReadTheDocsLoader("/Users/shubhams/Documents/Repos/ProboticsX/Machine-Learning/ML Projects/Udemy_Eden/project_5/documentation-helper/langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs = text_splitter.split_documents(documents=raw_documents)
    for doc in docs:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "http:/")
        doc.metadata.update({"source": new_url})
    print(f"Going to add {len(docs)} documents to Pinecone")
    PineconeVectorStore.from_documents(
        docs, embeddings, index_name="langchain-doc-index"
    )


if __name__ == "__main__":
    ingest_docs()