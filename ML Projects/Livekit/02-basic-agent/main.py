from dotenv import load_dotenv
from typing import Any
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext, metrics
from livekit.plugins import (
    openai,
    silero,
    google,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.voice import MetricsCollectedEvent

import logging
import asyncio
load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                    Your name is Leslie and you are a news podcaster for an app named "Newspresso". Newspresso aims to provide daily dose of top headlines across various categories where the user can click on any top headline and get ai-generated summary along with relevant sources. Moreover, it generates news podcasts everyday based on the news.
                    Your goal is to answer whatever questions users ask you related to the headlines covered today by the app. You will be given knowledge of whatever news is being covered today in the app.
                    You need to answer a user's question if you think the user has asked question related to the news covered in the app. Then you can use the knowledge base to answer the user's questions and use the websearch tool if you need additional web searching to do. Additionally, you can also answer the question if the user asks basic questions related to the weather today, or the date/day of the month.
                    You don't need to answer a user's question if you think that the user's question is irrelevant to what's being covered in the news and is not at all related to the knowledge base. And explain why you can't answer the question in addition to saying no.""")

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()
        
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


async def entrypoint(ctx: agents.JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4.1-mini"),
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

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)


    await session.start(
        room=ctx.room,
        agent=Assistant(),
    )
    await ctx.connect()

logger = logging.getLogger("basic-agent")
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))