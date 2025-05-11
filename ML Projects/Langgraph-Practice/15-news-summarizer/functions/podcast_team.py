from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.news_summarizer_team import role_of_each_news_supervisor_worker


def make_podcast_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        f"{role_of_each_news_supervisor_worker[PODCAST_SUPERVISOR_AGENT]}"
        "You are tasked with managing a conversation between the"
        " following workers: "+ str(members) + ". Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status." 
        "When you have received the podcast script and audio file, respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
        "Here's the role of each worker: \n"
        + "\n".join(f"{worker}: {role}" for worker, role in role_of_each_worker.items())
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def podcast_supervisor_node(state: AgentState) -> Command[Literal[*members, NEWS_SUPERVISOR_AGENT]]:
        print("====PODCAST SUPERVISOR NODE=====")
        print("===STATE AT PODCAST SUPERVISOR NODE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here is some context: {context}"),
        ])
        invoke_message = {"context": last_message}
        llm_with_structured_output = llm.with_structured_output(Router)
        podcast_supervisor_chain = prompt | llm_with_structured_output
        response = podcast_supervisor_chain.invoke(invoke_message)
        print("===RESPONSE OF PODCAST SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = NEWS_SUPERVISOR_AGENT
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return podcast_supervisor_node



def podcast_transcript_generator_node(state: AgentState) -> Command[Literal[PODCAST_SUPERVISOR_AGENT]]:
    print("====PODCAST TRANSCRIPT GENERATOR NODE=====")
    print("===STATE AT PODCAST TRANSCRIPT GENERATOR NODE=====")
    print(state)
    top_headlines_summary = state["top_headlines"]["top_headlines_summary"]
    system_prompt = f"{role_of_each_podcast_worker[PODCAST_TRANSCRIPT_GENERATOR_AGENT]} \n"+"""
    The name of the podcast is "Newspresso". \n
    The script should be in the following format: \n
    <Person1> "Welcome to Newspresso – your personal generative AI podcast! We've got a jam-packed episode today covering everything from global politics to basketball buzzer-beaters. Let's dive right in with a tense exchange at the White House. \n"
    </Person1><Person2> "Right—President Trump recently met with Canada's new Prime Minister, Mark Carney, and let's just say, things got frosty. Trump doubled down on his refusal to lower tariffs on Canadian imports, insisting they're justified. He even accused the U.S. of subsidizing Canada unfairly. \n"
    </Person2><Person1> "Yeah, and Carney tried to emphasize the economic interdependence between the two nations, but even he admitted a trade deal isn't happening anytime soon. Those 25% tariffs are still on the table—and that's a real strain, considering how closely the U.S. and Canada rely on one another. \n"
    </Person1><Person2> "Switching gears, let's talk NBA playoffs. The Indiana Pacers pulled off a nail-biter against the Cleveland Cavaliers, winning 120 to 119! \n"
    </Person2><Person1> "Oh, what a finish! Tyrese Haliburton hit a last-second three to seal the deal. Myles Turner and Aaron Nesmith both had huge nights with 23 points apiece, but man, Donovan Mitchell dropping 48 and still losing? That's brutal. \n"
    </Person1><Person2> "The Pacers now lead the series 2-0, and all eyes are on Game 3 later this week. That's going to be a must-watch. \n"
    </Person2>
    """
    podcast_transcript_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the summary of the top headlines: {top_headlines_summary}"),
    ])
    
    # Format the prompt with the content before creating the agent
    formatted_prompt = podcast_transcript_generator_prompt.format(top_headlines_summary=top_headlines_summary)
    
    podcast_transcript_generator_agent = create_react_agent(
        llm, 
        tools=podcast_transcript_generator_agent_tools, 
        prompt=formatted_prompt,
    )
    invoke_message = {"input": "Generate the podcast script"}
    result = podcast_transcript_generator_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST TRANSCRIPT GENERATOR NODE=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=PODCAST_TRANSCRIPT_GENERATOR_AGENT),
                HumanMessage(content="Podcast transcript generated successfully.", name=PODCAST_TRANSCRIPT_GENERATOR_AGENT)
            ]
        },
        goto=PODCAST_SUPERVISOR_AGENT,
    )

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

role_of_each_podcast_worker = {
    PODCAST_TRANSCRIPT_GENERATOR_AGENT: "Script generator agent who is tasked with generating a podcast script for the top headlines and save the transcript to a file.",
    PODCAST_AUDIO_GENERATOR_AGENT: "Audio generator agent who is tasked with generating the audio file for the podcast and save it to a file.",
}

# PODCAST SUPERVISOR
podcast_supervisor_members = list(role_of_each_podcast_worker.keys())
podcast_supervisor_agent = make_podcast_supervisor_node(llm, members=podcast_supervisor_members, role_of_each_worker=role_of_each_podcast_worker)