import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from oauthlib.uri_validate import query
from openai import embeddings

load_dotenv()
if __name__=="__main__":
    pdf_path = '/Langchain/project_4/paper.pdf'
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    #chunking
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    #embedding and storing it in vector db
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings) #local vector store
    vectorstore.save_local("faiss_index_react")

    #use the vectorstore that was saved locally previously
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    query = "Give me a gist of ReAct in 3 sentences"
    res = retrieval_chain.invoke(input={"input": query})
    print(res["answer"])