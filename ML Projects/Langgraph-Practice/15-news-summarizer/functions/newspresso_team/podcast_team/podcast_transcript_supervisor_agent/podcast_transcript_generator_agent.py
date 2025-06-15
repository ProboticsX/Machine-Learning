from common_imports import *
from constants import *
from classes import AgentState, PodcastClass, PodcastSummaryAndTitleClass
from tools.helper_tools.tools import *
from functions.newspresso_team.podcast_team.podcast_transcript_supervisor_agent.podcast_transcript_supervisor_agent import role_of_each_podcast_transcript_supervisor_worker

def podcast_transcript_generator_node(state: AgentState) -> Command[Literal[PODCAST_TRANSCRIPT_SUPERVISOR_AGENT]]:
    print("====PODCAST TRANSCRIPT GENERATOR NODE=====")
    print("===STATE AT PODCAST TRANSCRIPT GENERATOR NODE=====")
    print(state)
    top_headlines_summary_json_file = state["top_headlines_class"]["top_headlines_summary_json_file"]
    date = state["category_class"]["date"]
    podcast_transcript_critique = ""
    current_podcast_transcript_critique_count = 0
    podcast_transcript = ""
    if state.get("podcast_class") is not None:
        if state.get("podcast_class").get("podcast_transcript") is not None:
            podcast_transcript = state["podcast_class"]["podcast_transcript"]
        if state.get("podcast_class").get("podcast_transcript_critique") is not None:
            podcast_transcript_critique = state["podcast_class"]["podcast_transcript_critique"]
        if state.get("podcast_class").get("podcast_transcript_critique_count") is not None:
            current_podcast_transcript_critique_count = state["podcast_class"]["podcast_transcript_critique_count"]
    system_prompt = f"{role_of_each_podcast_transcript_supervisor_worker[PODCAST_TRANSCRIPT_GENERATOR_AGENT]} \n"+"""
    The name of the podcast is "Newspresso". \n
    Please make sure to improve the script based on the critique of the podcast transcript and the existing podcast transcript (the path of the file is provided to you if it's existing). \n
    The script should be in the following format: \n
    Speaker 1: "Welcome to Newspresso – your personal generative AI podcast! We've got a jam-packed episode today covering everything from global politics to basketball buzzer-beaters. Let's dive right in with a tense exchange at the White House."
    Speaker 2: "Right—President Trump recently met with Canada's new Prime Minister, Mark Carney, and let's just say, things got frosty. Trump doubled down on his refusal to lower tariffs on Canadian imports, insisting they're justified. He even accused the U.S. of subsidizing Canada unfairly."
    Speaker 1: "Yeah, and Carney tried to emphasize the economic interdependence between the two nations, but even he admitted a trade deal isn't happening anytime soon. Those 25% tariffs are still on the table—and that's a real strain, considering how closely the U.S. and Canada rely on one another."
    Speaker 2: "Switching gears, let's talk NBA playoffs. The Indiana Pacers pulled off a nail-biter against the Cleveland Cavaliers, winning 120 to 119!"
    Speaker 1: "The Pacers now lead the series 2-0, and all eyes are on Game 3 later this week. That's going to be a must-watch."

    Make sure the podcasters cover the headlines in a detailed manner which will help build interest to the users listening. \n
    Don't include any special characters in the Speaker tags. Don't include any other tags in the script. It should strictly follow the format above. \n
    Finally, write the podcast transcript generated above to a file. Always save the new file over the existing file.
    """
    podcast_transcript_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the summary of the top headlines found in the json file for which you need to generate the podcast transcript for: {top_headlines_summary_json_file}. Here's the existing podcast transcript (if any) {podcast_transcript} and the critique of the existing podcast transcript (if any): {podcast_transcript_critique}. The generated podcast transcript should be saved here {podcast_transcript_file_path} and the date of the podcast is {date}."),
    ])
    podcast_transcript_file_path = podcast_transcript_file
    formatted_prompt = podcast_transcript_generator_prompt.format(top_headlines_summary_json_file=top_headlines_summary_json_file, podcast_transcript=podcast_transcript, podcast_transcript_critique=podcast_transcript_critique, podcast_transcript_file_path=podcast_transcript_file_path, date=date)
    podcast_transcript_generator_agent = create_react_agent(
        llm, 
        tools=transcript_generator_agent_tools, 
        prompt=formatted_prompt,
    )
    invoke_message = {"input": "Generate the podcast transcript"}
    result = podcast_transcript_generator_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST TRANSCRIPT GENERATOR NODE=====")
    print(result)
    podcast_class = PodcastClass(podcast_transcript_file_path=podcast_transcript_file_path)
    if state.get("podcast_class") is not None:
        podcast_class = state["podcast_class"].copy()
        podcast_class["podcast_transcript"] = result["messages"][-1].content
        podcast_class["podcast_transcript_file_path"] = podcast_transcript_file_path

    if current_podcast_transcript_critique_count < MAX_PODCAST_TRANSCRIPT_CRITIQUE_COUNT:
        return Command(
            update={
                "messages": [
                    HumanMessage(content="Please provide the critique based on the podcast transcript.", name=PODCAST_TRANSCRIPT_GENERATOR_AGENT)
                ],
                "podcast_class": podcast_class,
            },
            goto=PODCAST_TRANSCRIPT_CRITIC_AGENT,
        )
    
    system_prompt = """You are given a podcast transcript and you need to come up with a suitable summary of the transcript in about 100-150 words mentioning what the podcasters talk about.\n
                       Also, come up with a catchy title for the podcast in about 5-8 words. \n
    """
    summary_and_title_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the podcast transcript: {podcast_transcript}."),
    ])
    formatted_prompt = summary_and_title_generator_prompt.format(podcast_transcript=podcast_transcript)
    summary_and_title_generator_agent = create_react_agent(
        llm,
        tools=[],
        prompt=formatted_prompt,
        response_format=PodcastSummaryAndTitleClass
    )
    invoke_message = {"input": "Generate the summary and title of the podcast"}
    result = summary_and_title_generator_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST SUMMARY AND TITLE GENERATOR NODE=====")
    print(result)
    podcast_summary_and_title_class = PodcastSummaryAndTitleClass(podcast_summary=result["structured_response"]["podcast_summary"], podcast_title=result["structured_response"]["podcast_title"])
    podcast_class["podcast_summary_and_title_class"] = podcast_summary_and_title_class
    return Command(
        update={
            "messages": [
                HumanMessage(content="Podcast transcript generated and written to a file successfully.", name=PODCAST_TRANSCRIPT_GENERATOR_AGENT)
            ],
            "podcast_class": podcast_class,
        },
        goto=PODCAST_TRANSCRIPT_SUPERVISOR_AGENT,
    )