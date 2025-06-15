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
    system_prompt = f"{role_of_each_random_worker[PODCAST_TRANSCRIPT_CRITIC_AGENT]} \n"+ """
                    Please make sure that every dialogue is in the format of Speaker 1: "..." and Speaker 2: "...". \n
                    Don't replace Speaker 1 and Speaker 2 tags in the script.\n
                    Don't include any special characters or bold text in the Speaker tags. Don't include any other tags in the script. It should strictly follow the format above. \n
                    You will be given the podcast transcript file path. Please read the file and then provide the critique. \n
                    """
    # Introducing the Podcasters, Date and Day
    if current_podcast_transcript_critique_count == 1:
        system_prompt+= "Introduce the podcasters in the podcast show: Speaker 1 is Leslie and Speaker 2 is Marcus. Example of a good critique can be such as introducing the podcasters, mentioning the day and date \n"
    # Preventing bias and adding facts
    elif current_podcast_transcript_critique_count == 2:
        system_prompt+= "Example of a good critique can be such as preventing bias and adding facts related to the news! \n"
    # Add Human touch
    elif current_podcast_transcript_critique_count == 3:
        system_prompt+= """Example of a good critique can be adding human touch. \n
                            For eg: 
                            - Add elements where the podcasters are laughing
                            - Interrupting each other or talk over each other
                            - Asking each other questions, opinions \n"""
    # Make it engaging to the listeners
    elif current_podcast_transcript_critique_count == 4:
        system_prompt+= """Example of a good critique can be making the podcast more interesting to the users. \n
                            Make sure the some of the news headlines are covered in detail manner which can keep the users hooked. \n
                            Other ways to make the podcast engaging is as follows: \n
                            For eg: 
                            - Keep the conversation humurous and casual.
                            - Add fun facts related to the news being covered.
                            - Ask trivia questions to the users or from the fellow podcaster
                            - Interrupting each other or talk over each other
                            - Asking each other questions, opinions \n"""
    # Podcasters personalities
    elif current_podcast_transcript_critique_count == 5:
        system_prompt+= """Example of a good critique can be having the podcasters debate over a certain news topic in the news being covered. \n
                            It's important to have contrasting personalities for both of the podcasters so that the users can see both sides of the coin. \n
                            You can also define podcaster personas, for eg: \n
                            - If you are covering news related to Finance, then both Leslie and Marcus are experts in finance world and should speak like one. \n
                            - If you are covering news related to Entertainment, then Leslie can be a Hollywood fan girl and Marcus can be a novice guy who doesn't watch a lot of movies. """
    
    # Advertisement
    elif current_podcast_transcript_critique_count == 6:
        system_prompt+= """Example of a good critique can be telling the podcasters to promote an advertisement because that generates revenue for Newspresso. \n
                           The ad can be anything related to the category being covered in the news and should be kept short and precise. \n
                           Let the listeneres know it's a sponsorship or advertisement during the podcast."""
        
    podcast_transcript_critic_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the podcast transcript file path: {podcast_transcript_file_path}. The date is {date}"),
    ])
    formatted_prompt = podcast_transcript_critic_prompt.format(podcast_transcript_file_path=podcast_transcript_file_path, date=date)
    podcast_transcript_critic_react_agent = create_react_agent(
        llm,
        tools=podcast_transcript_critic_agent_tools,
        prompt=formatted_prompt,
    )
    invoke_message = {"input": "Please do the task as per the system prompt"}
    result = podcast_transcript_critic_react_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST TRANSCRIPT CRITIC NODE=====")
    print(result)

    current_podcast_transcript_critique_count+=1
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
