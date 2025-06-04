from dotenv import load_dotenv
from typing import Any
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext
from livekit.plugins import (
    openai,
    silero,
    google,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")

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

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=google.TTS(gender="female", voice_name="en-US-Chirp-HD-F"),
        vad=silero.VAD(session=None, opts=None).load(),
        turn_detection=MultilingualModel(),

    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        # allow_interruptions=True,
        # room_input_options=RoomInputOptions(
        #     # LiveKit Cloud enhanced noise cancellation
        #     # - If self-hosting, omit this parameter
        #     # - For telephony applications, use `BVCTelephony` for best results
        #     noise_cancellation=noise_cancellation.BVC(), 
        # ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))