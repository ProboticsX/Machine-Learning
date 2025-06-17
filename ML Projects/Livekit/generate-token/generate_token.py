from dotenv import load_dotenv
from typing import Any, List, Literal, Dict, Optional
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
from typing_extensions import TypedDict
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

from livekit import api
from flask import Flask

app = Flask(__name__)
ROOM_ID = "identity"
PARTICIPANT_ID = "my name"

@app.route('/getToken')
def getToken():
#   token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'),
#                         os.getenv('LIVEKIT_API_SECRET')) \
#     .with_identity("identity") \
#     .with_name("name") \
#     .with_grants(api.VideoGrants(
#         room_join=True,
#         room="my-room",
#         can_update_own_metadata=True)) \
#         .with_room_config(
#             api.RoomConfiguration(
#                 agents=[
#                     api.RoomAgentDispatch(
#                         agent_name="test-agent", metadata='{"selected_date": "2025-07-15"}'
#                     )
#                 ],
#             ),
#         )
  token = api.AccessToken(os.getenv('LIVEKIT_API_KEY'), os.getenv('LIVEKIT_API_SECRET')) \
    .with_identity(f"{PARTICIPANT_ID}") \
    .with_name(f"{ROOM_ID}") \
    .with_grants(api.VideoGrants(
        room_join=True,
        room="my-room",
        can_update_own_metadata=True
    ))
  token_str = token.to_jwt()
  print(f"Generated token: {token_str}")
  return token_str

if __name__ == '__main__':
    app.run(debug=True, port=5000)

