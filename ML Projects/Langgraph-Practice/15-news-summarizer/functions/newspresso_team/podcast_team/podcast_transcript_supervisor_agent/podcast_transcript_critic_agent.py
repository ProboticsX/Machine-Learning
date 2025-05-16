from common_imports import *
from constants import *
from classes import AgentState, PodcastClass
from tools.helper_tools.tools import *
from functions.helper_functions import role_of_each_random_worker

def podcast_transcript_critic_node(state: AgentState) -> Command[Literal[PODCAST_TRANSCRIPT_GENERATOR_AGENT]]:
    print("====PODCAST TRANSCRIPT CRITIC NODE=====")
    print("===STATE AT PODCAST TRANSCRIPT CRITIC NODE=====")
    print(state)

    podcast_transcript = state["podcast_class"]["podcast_transcript"]
    existing_podcast_transcript_critique = ""
    if state.get("podcast_class").get("podcast_transcript_critique") is not None:
        existing_podcast_transcript_critique = state["podcast_class"]["podcast_transcript_critique"]
    current_podcast_transcript_critique_count = 1
    if state.get("podcast_class").get("podcast_transcript_critique_count") is not None:
        current_podcast_transcript_critique_count = state["podcast_class"]["podcast_transcript_critique_count"]
    
    system_prompt = f"{role_of_each_random_worker[PODCAST_TRANSCRIPT_CRITIC_AGENT]} \n"+"""
     Please make sure to improve the critique based on the existing critique. \n
     Example of a good critique can be such as introducing the podcasters, mentioning the day and date, preventing bias, adding facts, adding human touch, etc. \n
     Introduce the podcasters in the podcast show: Person1 is Marcus and Person2 is Leslie. Please make sure to not use any other tags except <Person1> and <Person2>, don't replace these tags in the script.\n
     Make sure to not use any other tags like ---. Don't change the structure of the script.
    """
    podcast_transcript_critic_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the podcast transcript: {podcast_transcript} and the existing critique: {existing_podcast_transcript_critique}"),
    ])
    podcast_transcript_critic_chain = podcast_transcript_critic_prompt | llm
    invoke_message = {"podcast_transcript": podcast_transcript, "existing_podcast_transcript_critique": existing_podcast_transcript_critique}
    result = podcast_transcript_critic_chain.invoke(invoke_message)
    print("===RESULT OF PODCAST TRANSCRIPT CRITIC NODE=====")
    print(result)
    
    podcast_class = PodcastClass(
        podcast_transcript_critique=result.content,
        podcast_transcript_critique_count=current_podcast_transcript_critique_count
    )
    if state.get("podcast_class") is not None:
        podcast_class = state["podcast_class"].copy()
        podcast_class["podcast_transcript_critique"] = result.content
        podcast_class["podcast_transcript_critique_count"] = current_podcast_transcript_critique_count

    return Command(
        update={
            "messages": [
                HumanMessage(content="Podcast transcript critiqued successfully.", name=PODCAST_TRANSCRIPT_CRITIC_AGENT)
            ],
            "podcast_class": podcast_class,
        },
        goto=PODCAST_TRANSCRIPT_GENERATOR_AGENT,
    )
