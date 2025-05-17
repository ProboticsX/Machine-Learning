from common_imports import *
from constants import *
from classes import AgentState, PodcastClass
from tools.helper_tools.tools import *
from functions.newspresso_team.podcast_team.podcast_transcript_supervisor_agent.podcast_transcript_supervisor_agent import role_of_each_podcast_transcript_supervisor_worker

def podcast_transcript_generator_node(state: AgentState) -> Command[Literal[PODCAST_TRANSCRIPT_SUPERVISOR_AGENT]]:
    print("====PODCAST TRANSCRIPT GENERATOR NODE=====")
    print("===STATE AT PODCAST TRANSCRIPT GENERATOR NODE=====")
    print(state)
    top_headlines_summary = state["top_headlines"]["top_headlines_summary"]
    podcast_transcript_critique = ""
    existing_podcast_transcript = ""
    current_podcast_transcript_critique_count = 0
    if state.get("podcast_class") is not None:
        if state.get("podcast_class").get("podcast_transcript_critique") is not None:
            podcast_transcript_critique = state["podcast_class"]["podcast_transcript_critique"]
        if state.get("podcast_class").get("podcast_transcript") is not None:
            existing_podcast_transcript = state["podcast_class"]["podcast_transcript"]
        if state.get("podcast_class").get("podcast_transcript_critique_count") is not None:
            current_podcast_transcript_critique_count = state["podcast_class"]["podcast_transcript_critique_count"]
    system_prompt = f"{role_of_each_podcast_transcript_supervisor_worker[PODCAST_TRANSCRIPT_GENERATOR_AGENT]} \n"+"""
    The name of the podcast is "Newspresso". \n
    Please make sure to improve the script based on the critique of the podcast transcript and the existing podcast transcript. \n
    Please make sure to not use any other tags except <Person1> and <Person2>. \n
    The script should be in the following format: \n
    <Person1> "Welcome to Newspresso – your personal generative AI podcast! We've got a jam-packed episode today covering everything from global politics to basketball buzzer-beaters. Let's dive right in with a tense exchange at the White House." \n
    </Person1><Person2> "Right—President Trump recently met with Canada's new Prime Minister, Mark Carney, and let's just say, things got frosty. Trump doubled down on his refusal to lower tariffs on Canadian imports, insisting they're justified. He even accused the U.S. of subsidizing Canada unfairly." \n
    </Person2><Person1> "Yeah, and Carney tried to emphasize the economic interdependence between the two nations, but even he admitted a trade deal isn't happening anytime soon. Those 25% tariffs are still on the table—and that's a real strain, considering how closely the U.S. and Canada rely on one another." \n
    </Person1><Person2> "Switching gears, let's talk NBA playoffs. The Indiana Pacers pulled off a nail-biter against the Cleveland Cavaliers, winning 120 to 119!" \n
    </Person2><Person1> "Oh, what a finish! Tyrese Haliburton hit a last-second three to seal the deal. Myles Turner and Aaron Nesmith both had huge nights with 23 points apiece, but man, Donovan Mitchell dropping 48 and still losing? That's brutal." \n
    </Person1><Person2> "The Pacers now lead the series 2-0, and all eyes are on Game 3 later this week. That's going to be a must-watch." \n
    </Person2>\n

    Finally, write the podcast transcript generated above to a file.
    """
    podcast_transcript_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the summary of the top headlines: {top_headlines_summary} and the critique of the podcast transcript: {podcast_transcript_critique} and the existing podcast transcript: {existing_podcast_transcript}"),
    ])
    
    formatted_prompt = podcast_transcript_generator_prompt.format(top_headlines_summary=top_headlines_summary, podcast_transcript_critique=podcast_transcript_critique, existing_podcast_transcript=existing_podcast_transcript)
    podcast_transcript_generator_agent = create_react_agent(
        llm, 
        tools=transcript_generator_agent_tools, 
        prompt=formatted_prompt,
    )
    invoke_message = {"input": "Generate the podcast script"}
    result = podcast_transcript_generator_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST TRANSCRIPT GENERATOR NODE=====")
    print(result)
    podcast_class = PodcastClass(podcast_transcript=result["messages"][-1].content)
    if state.get("podcast_class") is not None:
        podcast_class = state["podcast_class"].copy()
        podcast_class["podcast_transcript"] = result["messages"][-1].content

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
    return Command(
        update={
            "messages": [
                HumanMessage(content="Podcast transcript generated and written to a file successfully.", name=PODCAST_TRANSCRIPT_GENERATOR_AGENT)
            ],
            "podcast_class": podcast_class,
        },
        goto=PODCAST_TRANSCRIPT_SUPERVISOR_AGENT,
    )