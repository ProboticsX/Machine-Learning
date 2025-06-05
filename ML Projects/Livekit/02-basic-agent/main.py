from dotenv import load_dotenv
from typing import Any
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext, metrics, UserStateChangedEvent, JobProcess
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


load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                    Your name is Leslie and you are a news podcaster for an app named "Newspresso". Newspresso aims to provide daily dose of top headlines across various categories where the user can click on any top headline and get ai-generated summary along with relevant sources. Moreover, it generates news podcasts everyday based on the news. \n
                    Your goal is to answer whatever questions users ask you related to the headlines covered today by the app. You will be given knowledge of whatever news is being covered today in the app. \n
                    Please don't answer any questions that's not covered in the news. \n
                    You need to answer a user's question if you think the user has asked question related to the news covered in the app. Then you can use the knowledge base to answer the user's questions and use the websearch tool if you need additional web searching to do. Additionally, you can also answer the question if the user asks basic questions related to the weather today, or the date/day of the month. \n
                    You don't need to answer a user's question if you think that the user's question is irrelevant to what's being covered in the news and is not at all related to the knowledge base. And explain why you can't answer the question in addition to saying no. \n""")

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        # self.session.generate_reply()
        self.session.say("Hey! I'm Leslie from Newspresso! How can I assist you today?")
        
    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict[str, Any]:
        """Look up weather information for a given location.
        
        Args:
            location: The location to look up weather information for.
        """

        return {"weather": "sunny", "temperature_f": 70}
    
    # to hang up the call as part of a function call
    @function_tool
    async def end_call(self, ctx: RunContext):
        """Use this tool when the user has signaled they wish to end the current call. The session will end automatically after invoking this tool."""
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()
        logger.info("Closing session from function tool")
        await self.session.generate_reply(instructions="say goodbye to the user")
        self._closing_task = asyncio.create_task(self.session.aclose())
    
    @function_tool
    async def get_todays_date(self) -> str:
        """Gets the today's date.
        Returns:
            str: The today's date in the format of YYYY-MM-DD.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        day = datetime.now().strftime("%A")
        print(f"get_todays_date: {today}")
        return f"{today}"

    
    @function_tool
    async def get_result(self, query: str, date: str, top_k: int = 1):
        """Retrieve relevant headlines based on the query.
        
        Args:
            query (str): The search query from the user
            top_k (int): Number of results to return (default: 1)
            date (str): The today's date in the format of YYYY-MM-DD.
            
        Returns:
            List[Dict[str, Any]]: List of retrieved headlines with their metadata""" 
        # date="2025-06-05"       
        query_embedding = await embeddings.aembed_query(query)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        headlines = []
        for match in results.matches:
            # Extract date from ID (format: date_category_headline_index)
            doc_date = match.id.split('_')[0]
            
            # Apply date filter if specified
            if date and doc_date != date:
                continue
                
            headlines.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
                "date": doc_date
            })
            
            # Break if we have enough results after filtering
            if len(headlines) >= top_k:
                break
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here is the user question: {query}. The documents: {headlines}. The date is {date}"),
        ])
        invoke_message = {"input": "Please do the task as per the system prompt"}
        formatted_prompt = prompt.format(query=query, headlines=headlines, date=date)
        rag_generator_agent = create_react_agent(llm, 
                                             tools=[],
                                             prompt = formatted_prompt,
                                             )
        result_from_agent = rag_generator_agent.invoke(invoke_message)
        print(f"get_result: {result_from_agent}")
        return result_from_agent['messages'][-1].content


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
    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    inactivity_task: asyncio.Task | None = None

    async def user_presence_task():
        # try to ping the user 3 times, if we get no answer, close the session
        for _ in range(3):
            await session.generate_reply(
                instructions=(
                    "The user has been inactive. Politely check if the user is still present, and "
                    "gently guide the conversation back toward your intended goal."
                )
            )
            await asyncio.sleep(10)

        await asyncio.shield(session.aclose())
        ctx.delete_room()

    @session.on("user_state_changed")
    def _user_state_changed(ev: UserStateChangedEvent):
        nonlocal inactivity_task
        if ev.new_state == "away":
            inactivity_task = asyncio.create_task(user_presence_task())
            return

        # ev.new_state: listening, speaking, ..
        if inactivity_task is not None:
            inactivity_task.cancel()

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)


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
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings()
system_prompt = """You are a helpful assistant that generates the final answer to the user question. Please don't answer the questions in case it's not relevant to the documents attached and let the user know about it."""

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