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
    podcast_transcript_file_path = podcast_transcript_file
    system_prompt = f"{role_of_each_podcast_worker[PODCAST_AUDIO_GENERATOR_AGENT]}."+"""
        Use the audio file path from the podcast audio just generated and push it to the firebase storage as per the category.
    """
    podcast_audio_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the podcast transcript file path: {podcast_transcript_file_path}, the category of the podcast: {category} and the output path for the podcast audio: {podcast_audio_dir}. The podcast audio file name should be {podcast_audio_file_name}."),
    ])
    formatted_prompt = podcast_audio_generator_prompt.format(podcast_transcript_file_path=podcast_transcript_file_path, category=category, podcast_audio_dir = podcast_audio_dir,podcast_audio_file_name=podcast_audio_file_name)
    podcast_audio_generator_agent = create_react_agent(llm, 
                                             tools=podcast_audio_generator_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Please do the task as per the system prompt"}
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