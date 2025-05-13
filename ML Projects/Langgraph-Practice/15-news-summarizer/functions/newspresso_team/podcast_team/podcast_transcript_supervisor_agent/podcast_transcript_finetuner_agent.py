from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.newspresso_team.podcast_team.podcast_transcript_supervisor_agent.podcast_transcript_supervisor_agent import role_of_each_podcast_transcript_supervisor_worker

def podcast_transcript_finetuner_node(state: AgentState) -> Command[Literal[PODCAST_TRANSCRIPT_SUPERVISOR_AGENT]]:
    print("====PODCAST TRANSCRIPT FINETUNER NODE=====")
    print("===STATE AT PODCAST TRANSCRIPT FINETUNER NODE=====")
    print(state)
    podcast_transcript = state["podcast_transcript"]
    system_prompt = f"{role_of_each_podcast_transcript_supervisor_worker[PODCAST_TRANSCRIPT_FINETUNER_AGENT]} \n"+"""
    You are tasked with finetuning the podcast transcript. Below are the instructions on how to do it: \n
     - Introduce the podcasters in the podcast show: Person1 is Marcus and Person2 is Leslie. Please make sure to not use any other tags except <Person1> and <Person2>, don't replace these tags in the script.\n
     - Make sure the podcasters mention Today's day and date at the start of the podcast show. \n
     - Add human touch to the script by adding interjections, pauses or when the person is thinking.\n
     - Add other natural human cues like "aah", "um", "like", "you know", etc or when a human repeats some words unintentionally like "yeah, yeah" or "like, like". Make sure to use these cues only 3-4 times in the script.\n
     - Make sure to not use any other tags like ---. Don't change the structure of the script.
    """
    podcast_transcript_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the podcast transcript: {podcast_transcript}"),
    ])
    
    # Format the prompt with the content before creating the agent
    formatted_prompt = podcast_transcript_generator_prompt.format(podcast_transcript=podcast_transcript)
    
    podcast_transcript_generator_agent = create_react_agent(
        llm, 
        tools=podcast_transcript_finetuner_agent_tools, 
        prompt=formatted_prompt,
    )
    invoke_message = {"input": "Finetune the podcast script"}
    result = podcast_transcript_generator_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST TRANSCRIPT FINETUNER NODE=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=PODCAST_TRANSCRIPT_FINETUNER_AGENT),
                HumanMessage(content="Podcast transcript finetuned successfully.", name=PODCAST_TRANSCRIPT_FINETUNER_AGENT)
            ],
            "podcast_transcript": result["messages"][-1].content
        },
        goto=PODCAST_TRANSCRIPT_SUPERVISOR_AGENT,
    )