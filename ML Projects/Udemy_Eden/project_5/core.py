from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "langchain-doc-index"

def run_llm(query: str):
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    #Create a chain with LLM and prompt: Generative part
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    # Combine the chain created above with the retrieval chain with the data from the vectorstore and generate a new chain
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    # Invoke relevant data
    result = retrieval_chain.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    res = run_llm("What is a Langchain chain?")
    print(res)