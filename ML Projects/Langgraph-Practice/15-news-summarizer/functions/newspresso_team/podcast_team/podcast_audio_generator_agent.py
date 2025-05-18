from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.newspresso_team.podcast_team.podcast_supervisor_agent import role_of_each_podcast_worker

def podcast_audio_generator_node(state: AgentState) -> Command[Literal[PODCAST_SUPERVISOR_AGENT]]:
    print("====PODCAST AUDIO GENERATOR NODE=====")
    print("===STATE AT PODCAST AUDIO GENERATOR NODE=====")
    print(state)
    category = state["category_class"]["category"]
    system_prompt = f"{role_of_each_podcast_worker[PODCAST_AUDIO_GENERATOR_AGENT]}."+"""
        Use the audio file path from the podcastaudio just generated and push it to the firebase storage as per the category.
    """
    podcast_audio_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the transcript file path: {transcript_file}, the category of the podcast: {category}"),
    ])
    formatted_prompt = podcast_audio_generator_prompt.format(transcript_file=transcript_file, category=category)
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
                HumanMessage(content="Podcast audio generated from the podcast transcript successfully! The audio file has been pushed to the firebase storage.", name=PODCAST_AUDIO_GENERATOR_AGENT)
            ],
        },
        goto=PODCAST_SUPERVISOR_AGENT,
    )