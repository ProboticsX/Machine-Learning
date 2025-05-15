from common_imports import *
from constants import *
from classes import AgentState
from tools.helper_tools.tools import *
from functions.newspresso_team.podcast_team.podcast_transcript_supervisor_agent.podcast_transcript_supervisor_agent import role_of_each_podcast_transcript_supervisor_worker

def podcast_transcript_writer_node(state: AgentState) -> Command[Literal[PODCAST_TRANSCRIPT_SUPERVISOR_AGENT]]:
    print("====PODCAST TRANSCRIPT WRITER NODE=====")
    print("===STATE AT PODCAST TRANSCRIPT WRITER NODE=====")
    print(state)
    podcast_transcript = state["podcast_class"]["podcast_transcript"]
    system_prompt = f"{role_of_each_podcast_transcript_supervisor_worker[PODCAST_TRANSCRIPT_WRITER_AGENT]}" + """
    Please make sure to use the tags <Person1> and <Person2> in the podcast transcript while writing it to a file.
    """
    podcast_transcript_generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the podcast transcript: {podcast_transcript}"),
    ])
    
    # Format the prompt with the content before creating the agent
    formatted_prompt = podcast_transcript_generator_prompt.format(podcast_transcript=podcast_transcript)
    
    podcast_transcript_generator_agent = create_react_agent(
        llm, 
        tools=podcast_transcript_writer_agent_tools, 
        prompt=formatted_prompt,
    )
    invoke_message = {"input": "Write the podcast transcript to a file"}
    result = podcast_transcript_generator_agent.invoke(invoke_message)
    print("===RESULT OF PODCAST TRANSCRIPT WRITER NODE=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=PODCAST_TRANSCRIPT_WRITER_AGENT),
                HumanMessage(content="Podcast transcript written successfully to a file. Now generate the audio file from the transcript.", name=PODCAST_TRANSCRIPT_WRITER_AGENT)
            ]
        },
        goto=PODCAST_TRANSCRIPT_SUPERVISOR_AGENT,
    )