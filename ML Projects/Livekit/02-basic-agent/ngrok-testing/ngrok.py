from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pyngrok import ngrok
import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import asyncio
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="News Query Processing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone and embeddings
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "newspresso"
dense_index = pc.Index(f"{index_name}-dense")
sparse_index = pc.Index(f"{index_name}-sparse")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4.1-mini")

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=1000)

# Initialize conversation memory store
conversation_memory = defaultdict(list)

# System prompts for RAG
rag_grader_system_prompt = """You are an agent who is tasked with grading the relevance of retrieved documents to a user question.
                            A document can be graded relevant in case the question asked is relevant to the document or if the subject/topic in the question is mentioned in the document. \n
                            You will be given a question, conversation history, and a document for reference. \n
                            The output should be a binary score of yes or no. \n
                            If the document is relevant to the question or if the question refers to previous context in the conversation, the output should be yes. \n
                            If the document is not relevant to the question or there are no documents retrieved or the document is not from today's date, the output should be no. \n
                            """

rag_generator_system_prompt = """You are a helpful assistant that generates the final answer to the user question. \n
                                You have access to the conversation history to understand context and references to previous questions. \n
                                Please don't answer the questions in case it's not relevant to the documents attached and let the user know about it. \n
                                You will be given the (yes/no) decision on whether question is answerable or not. \n
                                If the question is answerable, you will be given the filtered documents in case the question is answerable. \n               
                                If the question is not answerable, you need to respond in a polite manner to the user that the question cannot be answered.
                                """

newspresso_assistant_system_prompt = """Your name is Leslie and you are a news assistant for an app named "Newspresso". \n
                    Newspresso aims to provide daily dose of top headlines across various categories where the user can click on any top headline and get AI-generated summary along with relevant sources. \n
                    Moreover, it generates news podcasts everyday based on the news. \n
                    Your goal is to answer whatever questions users ask you related to the headlines covered today by the app.\n
                    You have access to the conversation history to understand context and references to previous questions. \n
                    Don't use general knowledge to answer any of the user's question on your own. \n
                    You need to only provide the answer to the question if it is answerable.
                    If the question is not answerable, you need to respond in a polite manner to the user that the question cannot be answered.
                    """

class QueryRequest(BaseModel):
    query: str
    user_id: str = None  # Optional user identifier
    conversation_id: str = None  # To identify different conversation threads

class QueryResponse(BaseModel):
    response: str
    status: str

async def get_todays_date() -> str:
    """Gets today's date."""
    return datetime.now().strftime("%Y-%m-%d")

async def create_sparse_vector(text: str) -> Dict[str, List]:
    """Create a sparse vector using TF-IDF."""
    tfidf_matrix = tfidf.fit_transform([text])
    feature_names = tfidf.get_feature_names_out()
    values = tfidf_matrix.toarray()[0]
    
    indices = []
    vector_values = []
    
    for i, value in enumerate(values):
        if value > 0:
            indices.append(i)
            vector_values.append(float(value))
    
    return {
        "indices": indices,
        "values": vector_values
    }

async def get_relevant_headlines(query: str, top_k: int, date: str) -> List[Dict[str, Any]]:
    """Retrieve relevant headlines using hybrid search."""
    dense_embedding = await embeddings.aembed_query(query)
    sparse_vector = await create_sparse_vector(query)
    pinecone_filter = {"published_at": {"$eq": date}} if date else None

    dense_results = dense_index.query(
        vector=dense_embedding,
        top_k=top_k * 2,
        include_metadata=True,
        filter=pinecone_filter
    )

    sparse_results = sparse_index.query(
        sparse_vector=sparse_vector,
        top_k=top_k * 2,
        include_metadata=True,
        filter=pinecone_filter
    )

    # Merge and deduplicate results
    unique_results = {}
    
    for match in dense_results.matches:
        unique_results[match.id] = {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata
        }
    
    for match in sparse_results.matches:
        if match.id in unique_results:
            unique_results[match.id]["score"] = max(unique_results[match.id]["score"], match.score)
        else:
            unique_results[match.id] = {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }

    merged_results = list(unique_results.values())
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Rerank documents
    rerank_docs = []
    for r in merged_results:
        text_content = r["metadata"].get("chunk_text", "")
        if text_content:
            rerank_docs.append({
                "text": text_content,
                "id": r["id"]
            })
    
    if not rerank_docs:
        return merged_results[:top_k]
        
    try:
        rerank_result = pc.inference.rerank(
            model="bge-reranker-v2-m3",
            query=query,
            documents=rerank_docs,
            top_n=min(top_k, len(rerank_docs)),
            return_documents=True
        )
        
        id_to_result = {r["id"]: r for r in merged_results}
        reranked = []
        
        for row in rerank_result.data:
            doc_id = row["document"]["id"]
            if doc_id in id_to_result:
                result = {
                    "id": doc_id,
                    "score": id_to_result[doc_id]["score"],
                    "rerank_score": row["score"],
                    "metadata": id_to_result[doc_id]["metadata"]
                }
                reranked.append(result)
        
        return reranked[:top_k]
        
    except Exception as e:
        print(f"Error during reranking: {e}")
        return merged_results[:top_k]

async def stream_agent_response(agent, input_text: str) -> AsyncGenerator[str, None]:
    """Stream the response from an agent."""
    result = agent.invoke({"input": input_text})
    # Stream the response word by word
    words = result['messages'][-1].content.split()
    for word in words:
        yield f"{word} "
        await asyncio.sleep(0.05)  # Add a small delay between words

@app.post("/process_query")
async def process_query(request: QueryRequest):
    async def generate_stream():
        try:
            query = request.query
            conversation_id = request.conversation_id or "default"
            
            # Get conversation history
            chat_history = conversation_memory[conversation_id]
            formatted_history = "\n".join([f"User: {msg['query']}\nAssistant: {msg['response']}" for msg in chat_history[-5:]])  # Keep last 5 exchanges
            
            date = await get_todays_date()
            headlines = await get_relevant_headlines(query=query, top_k=2, date=date)
            
            # RAG Grader Agent
            prompt = ChatPromptTemplate.from_messages([
                ("system", rag_grader_system_prompt),
                ("user", "Here is the conversation history:\n{history}\n\nCurrent user question: {query}\nThe document to be graded: {document}."),
            ])
            
            filtered_docs = []
            for document in headlines:
                formatted_prompt = prompt.format(
                    query=query,
                    document=document['metadata']['chunk_text'],
                    history=formatted_history
                )
                rag_grader_agent = create_react_agent(llm, 
                                                    tools=[],
                                                    prompt=formatted_prompt
                                                    )
                document_result = rag_grader_agent.invoke({"input": "Please do the task as per the system prompt"})
                
                if document_result["messages"][-1].content.lower() == "yes" or document["rerank_score"] > 0.5:
                    filtered_docs.append(document)
            
            rag_grader_final_result = "yes" if filtered_docs else "no"
            
            # RAG Generator Agent
            prompt = ChatPromptTemplate.from_messages([
                ("system", rag_generator_system_prompt),
                ("user", "Here is the conversation history:\n{history}\n\nCurrent user question: {query}\nResult from grader whether the question asked is relevant to the news or not (Yes/No)? : {rag_grader_final_result}\nThe filtered documents (in case the question is answerable): {filtered_docs}."),
            ])
            
            formatted_prompt = prompt.format(
                query=query,
                rag_grader_final_result=rag_grader_final_result,
                filtered_docs=filtered_docs,
                history=formatted_history
            )
            
            rag_generator_agent = create_react_agent(llm, 
                                                 tools=[],
                                                 prompt=formatted_prompt
                                                 )
            
            # Get RAG Generator response
            rag_generator_response = rag_generator_agent.invoke({"input": "Please do the task as per the system prompt"})['messages'][-1].content
            
            # Newspresso Assistant Agent
            prompt = ChatPromptTemplate.from_messages([
                ("system", newspresso_assistant_system_prompt),
                ("user", "Here is the conversation history:\n{history}\n\nCurrent user question: {query}\nThe result from the RAG Generator Agent: {result}."),
            ])
            
            formatted_prompt = prompt.format(
                query=query,
                result=rag_generator_response,
                history=formatted_history
            )
            
            newspresso_assistant_agent = create_react_agent(llm, 
                                                        tools=[],
                                                        prompt=formatted_prompt
                                                        )
            
            # Get the final response
            final_response = newspresso_assistant_agent.invoke({"input": "Please do the task as per the system prompt"})['messages'][-1].content
            
            # Store the conversation
            conversation_memory[conversation_id].append({
                "query": query,
                "response": final_response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Stream the response word by word
            words = final_response.split()
            for i, word in enumerate(words):
                response = {
                    "response": word + (" " if i < len(words) - 1 else ""),
                    "status": "streaming" if i < len(words) - 1 else "success",
                    "conversation_id": conversation_id
                }
                yield json.dumps(response) + "\n"
                await asyncio.sleep(0.1)
            
        except Exception as e:
            error_response = {
                "response": f"Error: {str(e)}",
                "status": "error",
                "conversation_id": conversation_id if 'conversation_id' in locals() else None
            }
            yield json.dumps(error_response) + "\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson"  # Use newline-delimited JSON
    )

def start_ngrok():
    # Open a ngrok tunnel to the HTTP server
    public_url = ngrok.connect(8000).public_url
    print(f" * ngrok tunnel \"{public_url}\" -> http://127.0.0.1:8000")
    return public_url

if __name__ == "__main__":
    # Start ngrok
    public_url = start_ngrok()
    
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
