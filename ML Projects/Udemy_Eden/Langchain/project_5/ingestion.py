import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader, FireCrawlLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import embeddings

load_dotenv()
embeddings = OpenAIEmbeddings()

def ingest_docs():
    loader = ReadTheDocsLoader(
        "/Langchain/project_5/documentation-helper/langchain-docs/api.python.langchain.com/en/latest")
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

def ingest_firecrawl():
    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/integrations/chat//",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/chat_loaders/",
        "https://python.langchain.com/docs/concepts/",
    ]
    langchain_documents_base_urls2 = [langchain_documents_base_urls[0]]
    for url in langchain_documents_base_urls2:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url = url,
            mode = "crawl",
            params={
                "pageOptions": {"onlyMainContent": True},
            },
        )
        docs = loader.load()
        # No need for Text Splitting with Firecrawl
        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )


if __name__ == "__main__":
    # ingest_docs()
    ingest_firecrawl()