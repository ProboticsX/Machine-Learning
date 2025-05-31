from common_imports import *
from constants import *
from classes import AgentState, PodcastClass
from tools.helper_tools.tools import *
from functions.helper_functions import role_of_each_random_worker

def podcast_transcript_critic_node(state: AgentState) -> Command[Literal[PODCAST_TRANSCRIPT_GENERATOR_AGENT]]:
    print("====PODCAST TRANSCRIPT CRITIC NODE=====")
    print("===STATE AT PODCAST TRANSCRIPT CRITIC NODE=====")
    print(state)
    date = state["category_class"]["date"]
    podcast_transcript_file_path = state["podcast_class"]["podcast_transcript_file_path"]
    existing_podcast_transcript_critique = ""
    if state.get("podcast_class").get("podcast_transcript_critique") is not None:
        existing_podcast_transcript_critique = state["podcast_class"]["podcast_transcript_critique"]
    current_podcast_transcript_critique_count = 1
    if state.get("podcast_class").get("podcast_transcript_critique_count") is not None:
        current_podcast_transcript_critique_count = state["podcast_class"]["podcast_transcript_critique_count"]
    
    system_prompt = f"{role_of_each_random_worker[PODCAST_TRANSCRIPT_CRITIC_AGENT]} \n"+"""
     Please make sure to improve the critique based on the existing critique. \n
     Example of a good critique can be such as introducing the podcasters, mentioning the day and date, preventing bias, adding facts, adding human touch, etc. \n
     Introduce the podcasters in the podcast show: Speaker 1 is Leslie and Speaker 2 is Marcus. Please make sure that every dialogue is in the format of Speaker 1: "..." and Speaker 2: "...". \n
     Don't replace Speaker 1 and Speaker 2 tags in the script.\n
     Make the podcast transcript more engaging and interesting to listen to. The podcasters can laugh and joke around related to the news. Moreover, they can interrupt each other and talk over each other. \n
     The podcasters can also talk about the news in a more casual and conversational manner. They can also mention a fun fact related to the news (if any). \n
     You will be given the podcast transcript file path. Please read the file and then provide the critique. \n
     Don't include any special characters or bold text in the Speaker tags. Don't include any other tags in the script. It should strictly follow the format above. \n
    """
    podcast_transcript_critic_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the podcast transcript file path: {podcast_transcript_file_path} and the existing critique: {existing_podcast_transcript_critique}. The date is {date}"),
    ])
    formatted_prompt = podcast_transcript_critic_prompt.format(podcast_transcript_file_path=podcast_transcript_file_path, existing_podcast_transcript_critique=existing_podcast_transcript_critique, date=date)
    podcast_transcript_critic_react_agent = create_react_agent(
        llm,
        tools=podcast_transcript_critic_agent_tools,
        prompt=formatted_prompt,
    )
    invoke_message = {"input": "Please do the task as per the system prompt"}
    result = podcast_transcript_critic_react_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST TRANSCRIPT CRITIC NODE=====")
    print(result)
    
    podcast_class = PodcastClass(
        podcast_transcript_critique=result["messages"][-1].content,
        podcast_transcript_critique_count=current_podcast_transcript_critique_count
    )
    if state.get("podcast_class") is not None:
        podcast_class = state["podcast_class"].copy()
        podcast_class["podcast_transcript_critique"] = result["messages"][-1].content
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
