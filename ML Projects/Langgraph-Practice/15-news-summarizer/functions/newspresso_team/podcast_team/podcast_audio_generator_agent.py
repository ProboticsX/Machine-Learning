from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.newspresso_team.podcast_team.podcast_supervisor_agent import role_of_each_podcast_worker

def podcast_audio_generator_node(state: AgentState) -> Command[Literal[PODCAST_SUPERVISOR_AGENT]]:
    print("====PODCAST AUDIO GENERATOR NODE=====")
    print("===STATE AT PODCAST AUDIO GENERATOR NODE=====")
    print(state)
    system_prompt = f"{role_of_each_podcast_worker[PODCAST_AUDIO_GENERATOR_AGENT]}."
    podcast_audio_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the transcript file path: {transcript_file}"),
    ])
    formatted_prompt = podcast_audio_generator_prompt.format(transcript_file=transcript_file)
    podcast_audio_generator_agent = create_react_agent(llm, 
                                             tools=podcast_audio_generator_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Generate the podcast audio from the transcript file"}
    result = podcast_audio_generator_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST AUDIO GENERATOR NODE=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=PODCAST_AUDIO_GENERATOR_AGENT),
                HumanMessage(content="Podcast audio generated from the transcript successfully.", name=PODCAST_AUDIO_GENERATOR_AGENT)
            ],
        },
        goto=PODCAST_SUPERVISOR_AGENT,
    )