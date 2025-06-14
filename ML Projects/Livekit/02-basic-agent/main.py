from dotenv import load_dotenv
from typing import Any, List, Literal, Dict, Optional
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext, metrics, UserStateChangedEvent, JobProcess, UserInputTranscribedEvent
from livekit.plugins import (
    openai,
    silero,
    google,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.voice import MetricsCollectedEvent
from pinecone import Pinecone
import os
import logging
import asyncio
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent, ToolNode
from datetime import datetime
from typing_extensions import TypedDict
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                    Your name is Leslie and you are a news podcaster for an app named "Newspresso". \n
                    Newspresso aims to provide daily dose of top headlines across various categories where the user can click on any top headline and get AI-generated summary along with relevant sources. \n
                    Moreover, it generates news podcasts everyday based on the news. \n
                    Your goal is to answer whatever questions users ask you related to the headlines covered today by the app.\n
                    Always use the tool attached (get_answer_from_user_query) to answer the user's question. Don't use general knowledge to answer any of the user's question on your own. \n
                    You need to only provide the answer to the question if it is answerable.
                    If the question is not answerable, you need to respond in a polite manner to the user that the question cannot be answered.
                    """)
        self.tfidf = TfidfVectorizer(max_features=1000)

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        # self.session.generate_reply()
        self.session.say("Hey! I'm Leslie from Newspresso! How can I assist you today?")
    
    # to hang up the call as part of a function call
    # @function_tool
    # async def end_call(self, ctx: RunContext):
    #     """Use this tool when the user has signaled they wish to end the current call. The session will end automatically after invoking this tool."""
    #     current_speech = ctx.session.current_speech
    #     if current_speech:
    #         await current_speech.wait_for_playout()
    #     logger.info("Closing session from function tool")
    #     await self.session.generate_reply(instructions="say goodbye to the user")
    #     self._closing_task = asyncio.create_task(self.session.aclose())
    
    @function_tool
    async def get_todays_date(self) -> str:
        """Gets the today's date.
        Returns:
            str: The today's date in the format of YYYY-MM-DD.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"get_todays_date: {today}")
        return f"{today}"

    @function_tool
    async def get_answer_from_user_query(self, user_transcript: str) -> str:
        """Use this tool to answer any user's question.
        
        Args:
            user_transcript (str): The user_transcript what has been asked by the user.
            
        Returns:
            str: Answer to the user's question""" 
        query = user_transcript
        date = await self.get_todays_date()
        headlines = await self.get_relevant_headlines(query=query, top_k=1, date=date)
        
        # RAG Grader Agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", rag_grader_system_prompt),
            ("user", "Here is the user question: {query}. The document to be graded: {document}."),
        ])
        invoke_message = {"input": "Please do the task as per the system prompt"}
        filtered_docs = []
        documents = headlines
        for document in documents:
            formatted_prompt = prompt.format(query=query, document=document['metadata']['chunk_text'])
            rag_grader_agent = create_react_agent(llm, 
                                                    tools=[],
                                                    prompt = formatted_prompt
                                                    )
            document_result = rag_grader_agent.invoke(invoke_message)
            print("===HERE IS THE RESULT OF THE DOCUMENT:", document['id'],"===")
            print(document_result)
            if document_result["messages"][-1].content.lower() == "yes" or document["rerank_score"] > 0.5:
                print("=====GRADER: DOCUMENT IS RELEVANT======")
                filtered_docs.append(document)
            else:
                print("=====GRADER: DOCUMENT IS NOT RELEVANT======")
        rag_grader_final_result = "yes"
        if len(filtered_docs) == 0:
            rag_grader_final_result = "no"
        print(f"rag_grader_final_result{rag_grader_final_result} and filtered documents: {filtered_docs}")
        
        # RAG Generator Agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", rag_generator_system_prompt),
            ("user", "Here is the user question: {query}. Result from grader whether the question asked is relevant to the news or not (Yes/No)? : {rag_grader_final_result} The filtered documents (in case the question is answerable): {filtered_docs}."),
        ])
        formatted_prompt = prompt.format(query=query, rag_grader_final_result=rag_grader_final_result,  filtered_docs=filtered_docs)
        rag_generator_agent = create_react_agent(llm, 
                                             tools=[self.get_perplexity_response],
                                             prompt = formatted_prompt,
                                             )
        result_from_rag_generator_agent = rag_generator_agent.invoke(invoke_message)
        print(f"get_answer_from_user_query: {result_from_rag_generator_agent}")
        return result_from_rag_generator_agent['messages'][-1].content


    #Normal Tool with async
    async def get_relevant_headlines(self, query: str, top_k: int, date: str) -> List[Dict[str, Any]]:
        """Retrieve relevant headlines using hybrid search.
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            date (str, optional): Filter results by date (format: YYYY-MM-DD)
            
        Returns:
            List[Dict[str, Any]]: List of retrieved headlines with their metadata
        """
        print(f"{query}, {top_k}, {date}")
        dense_embedding = await embeddings.aembed_query(query)
        sparse_vector = await self._create_sparse_vector(query)
        pinecone_filter = {"published_at": {"$eq": date}} if date else None

        dense_results = dense_index.query(
            vector=dense_embedding,
            top_k=top_k * 2,  # Fetch more results to account for filtering
            include_metadata=True,
            filter=pinecone_filter
        )
        print(f"dense_results:{dense_results}")

        sparse_results = sparse_index.query(
            sparse_vector=sparse_vector,
            top_k=top_k * 2,  # Fetch more results to account for filtering
            include_metadata=True,
            filter=pinecone_filter
        )
        print(f"sparse_results:{sparse_results}")

        merged_results = await self._merge_and_deduplicate_results(dense_results, sparse_results, query, top_k * 2)
        print(f"The results returned from RAG are:{merged_results}")

        return merged_results
    
    async def _create_sparse_vector(self, text: str) -> Dict[str, List]:
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
    
    async def _merge_and_deduplicate_results(self, dense_results, sparse_results, query: str, top_k: int) -> List[Dict[str, Any]]:
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
            rerank_result = pc.inference.rerank(
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

    #Normal Tool not needing async
    def get_perplexity_response(self, question: str):
        """Use this tool only when the question is answerable and you need extra information from web search. Gets the response from the perplexity API.
        Args:
            question: The question to ask the perplexity API.
        Returns:
            str: The response from the perplexity API.
        """

        payload = self.get_perplexity_payload(question)
        headers = self.get_perplexity_headers()
        response = requests.request("POST", PERPLEXITY_API_URL, json=payload, headers=headers)
        result = response.json()
        print(f"get_perplexity_response: {result}")
        return result

    def get_perplexity_payload(self, question: str):
        payload = {
            "model": perplexity_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers questions and provides information."},
                {"role": "user", "content": question}
            ],
        }
        return payload

    def get_perplexity_headers(self):
        headers = {
            "Authorization": f"Bearer {os.getenv("PERPLEXITY_API_KEY")}",
            "Content-Type": "application/json"
        }
        return headers

async def entrypoint(ctx: agents.JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4.1-mini"), #openai.LLM(model="gpt-4.1-mini"), #openai.realtime.RealtimeModel(),
        tts=google.TTS(gender="female", voice_name="en-US-Chirp-HD-F"),
        vad=silero.VAD(session=None, opts=None).load(),
        turn_detection=MultilingualModel(),

    )

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        print(f"User input transcribed: {event.transcript}, final: {event.is_final}")
    # log metrics as they are emitted, and total usage after session is over
    # usage_collector = metrics.UsageCollector()

    # @session.on("metrics_collected")
    # def _on_metrics_collected(ev: MetricsCollectedEvent):
    #     metrics.log_metrics(ev.metrics)
    #     usage_collector.collect(ev.metrics)

    # async def log_usage():
    #     summary = usage_collector.get_summary()
    #     logger.info(f"Usage: {summary}")

    # inactivity_task: asyncio.Task | None = None

    # async def user_presence_task():
    #     # try to ping the user 3 times, if we get no answer, close the session
    #     for _ in range(3):
    #         await session.generate_reply(
    #             instructions=(
    #                 "The user has been inactive. Politely check if the user is still present, and "
    #                 "gently guide the conversation back toward your intended goal."
    #             )
    #         )
    #         await asyncio.sleep(10)

    #     await asyncio.shield(session.aclose())
    #     ctx.delete_room()

    # @session.on("user_state_changed")
    # def _user_state_changed(ev: UserStateChangedEvent):
    #     nonlocal inactivity_task
    #     if ev.new_state == "away":
    #         inactivity_task = asyncio.create_task(user_presence_task())
    #         return

    #     # ev.new_state: listening, speaking, ..
    #     if inactivity_task is not None:
    #         inactivity_task.cancel()

    # shutdown callbacks are triggered when the session is over
    # ctx.add_shutdown_callback()


    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )
    await ctx.connect()

def prewarm_fnc(proc: JobProcess):
    pass


logger = logging.getLogger("basic-agent")
logger.setLevel(logging.INFO)
llm = ChatOpenAI(model_name="gpt-4.1-mini")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "newspresso"
dense_index = pc.Index(f"{index_name}-dense")
sparse_index = pc.Index(f"{index_name}-sparse")
embeddings = OpenAIEmbeddings()
rag_generator_system_prompt = """You are a helpful assistant that generates the final answer to the user question. \n
                                Please don't answer the questions in case it's not relevant to the documents attached and let the user know about it. \n
                                You will be given the (yes/no) decision on whether question is answerable or not. \n
                                If the question is answerable, you will be given the filtered documents in case the question is answerable and you can use the websearch tool in case you need more information.\n               
                                If the question is not answerable, you need to respond in a polite manner to the user that the question cannot be answered.
                                """
rag_grader_system_prompt = """You are an agent who is tasked with grading the relevance of retrieved documents to a user question.
                            A document can be graded relevant in case the question asked is relevant to the document or if the subject/topic in the question is mentioned in the document. \n
                            You will be given a question and a document for reference. \n
                            The output should be a binary score of yes or no. \n
                            If the document is relevant to the question, the output should be yes. \n
                            If the document is not relevant to the question or there are no documents retrieved or the document is not from today's date, the output should be no. \n
                            """
perplexity_model = "sonar"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm_fnc))
    # query = "What's the news related to stock market?"
    # headlines = "The U.S. stock market is navigating through a challenging but potentially rewarding landscape in 2025. According to Morgan Stanley, U.S. equities are expected to outperform international counterparts, with the S&P 500 forecasted to reach 6,500 by mid-2026, driven by improved earnings and potential monetary easing. Morningstar suggests that the U.S. stock market is trading at a discount, advocating for a market-weight stance with a focus on value and core sectors as valuations rise. Despite volatility, analysts expect positive earnings growth, though interest-rate dynamics may impact stock valuations."""
    # prompt = ChatPromptTemplate.from_messages([
    #         ("system", system_prompt),
    #         ("user", "Here is the user question: {query}. The documents: {headlines}."),
    #     ])
    # invoke_message = {"input": "Please do the task as per the system prompt"}
    # formatted_prompt = prompt.format(query=query, headlines=headlines)
    # rag_generator_agent = create_react_agent(llm, 
    #                                          tools=[],
    #                                          prompt = formatted_prompt,
    #                                          )
    # result_from_agent = rag_generator_agent.invoke(invoke_message)
    # print(f"get_result: {result_from_agent}")