from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="news-summarizer.png")

@tool
def multiply(a, b):
    """Multiply the two numbers"""
    return (a * b)

@tool
def add(a, b):
    """Add the two numbers"""
    return (a + b)

@tool
def divide(a, b):
    """Divide the two numbers"""
    return (a / b)

@tool
def get_stock_price(ticker: str) -> float:
    """Gets a stock price from Yahoo Finance.
    Args:
        ticker: ticker str
    """
    stock = yf.Ticker(ticker)
    return stock.info['previousClose']

@tool
def get_top_headlines() -> str:
    """Gets the top headlines from the news API."""
    news_tool = NewsContentTool()
    result = news_tool.get_top_headlines()
    return result

@tool
def write_to_file(content: str) -> str:
    """Writes the content to a file."""
    with open("podcast-script.txt", "w") as f:
        f.write(content)
    return "File written successfully"

def make_news_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
        "Here's the role of each worker: \n"
        f"{role_of_each_worker}"
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def news_supervisor_node(state: AgentState) -> Command[Literal[*members, END]]:
        print("====NEWS SUPERVISOR NODE=====")
        print("===STATE AT NEWS SUPERVISOR NODE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        invoke_message = [{"role": "system", "content": system_prompt},] + [last_message]
        llm_with_structured_output = llm.with_structured_output(Router)
        response = llm_with_structured_output.invoke(invoke_message)
        print("===RESPONSE OF NEWS SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return news_supervisor_node


def make_top_headlines_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        f"{role_of_each_news_supervisor_worker[TOP_HEADLINES_SUPERVISOR_AGENT]}"
        "You are tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status." 
        "When you have received the summary of the top headlines, respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
        "Here's the role of each worker: \n"
        f"{role_of_each_worker}"
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def top_headlines_supervisor_node(state: AgentState) -> Command[Literal[*members, NEWS_SUPERVISOR_AGENT]]:
        print("====TOP HEADLINES SUPERVISOR NODE=====")
        print("===STATE AT TOP HEADLINES SUPERVISOR NODE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        invoke_message = [{"role": "system", "content": system_prompt},] + [last_message]
        llm_with_structured_output = llm.with_structured_output(Router)
        response = llm_with_structured_output.invoke(invoke_message)
        print("===RESPONSE OF TOP HEADLINES SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = NEWS_SUPERVISOR_AGENT
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return top_headlines_supervisor_node

def make_podcast_supervisor_node(llm, members, role_of_each_worker) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        f"{role_of_each_news_supervisor_worker[PODCAST_SUPERVISOR_AGENT]}"
        "You are tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status." 
        "When you have received the podcast script, respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
        "Here's the role of each worker: \n"
        f"{role_of_each_worker}"
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
        invoke_message = [{"role": "system", "content": system_prompt},] + [last_message]
        llm_with_structured_output = llm.with_structured_output(Router)
        response = llm_with_structured_output.invoke(invoke_message)
        print("===RESPONSE OF PODCAST SUPERVISOR NODE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = NEWS_SUPERVISOR_AGENT
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return podcast_supervisor_node



def top_headlines_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES NODE=====")
    print("===STATE AT TOP HEADLINES NODE=====")
    print(state)
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_AGENT]}"
    top_headlines_agent = create_react_agent(llm, 
                                             tools=top_headlines_agent_tools, 
                                             prompt = system_prompt)
    result = top_headlines_agent.invoke(state)
    print("===RESULT OF TOP HEADLINES NODE=====")
    print(result)
    tool_message = result["messages"][-2]
    current_top_headlines = TopHeadlinesClass(
        top_headlines_full_content_from_tool=tool_message.content
    )
    if state.get("top_headlines") is not None:
        current_top_headlines = state["top_headlines"].copy()
        current_top_headlines["top_headlines_full_content_from_tool"] = tool_message.content
    return Command(
        update={
            "messages": [
                HumanMessage(content=tool_message.content, name=TOP_HEADLINES_AGENT)
            ],
            "top_headlines": current_top_headlines,
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )

def top_headlines_summarizer_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES SUMMARIZER NODE=====")
    print("===STATE AT TOP HEADLINES SUMMARIZER NODE=====")
    print(state)
    top_headlines_full_content_from_tool = state["top_headlines"]["top_headlines_full_content_from_tool"]
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_SUMMARIZER_AGENT]}. The summary should be in about 200-300 words for each headline."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the top headlines: {top_headlines_full_content_from_tool}"),
    ])
    top_headlines_summarizer_chain = prompt | llm
    invoke_message = {"top_headlines_full_content_from_tool": top_headlines_full_content_from_tool}
    result = top_headlines_summarizer_chain.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES SUMMARIZER NODE=====")
    print(result)
    current_top_headlines = TopHeadlinesClass(
            top_headlines_summary=result.content
        )
    if state.get("top_headlines") is not None:
        current_top_headlines = state["top_headlines"].copy()
        current_top_headlines["top_headlines_summary"] = result.content
        
    return Command(
        update={
            "messages": [
                HumanMessage(content=result.content, name=TOP_HEADLINES_SUMMARIZER_AGENT)
            ],
            "top_headlines": current_top_headlines,
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )

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
                HumanMessage(content=result["messages"][-1].content, name=PODCAST_TRANSCRIPT_GENERATOR_AGENT)
            ]
        },
        goto=PODCAST_SUPERVISOR_AGENT,
    )

# def podcast_audio_generator_node(state: AgentState) -> Command[Literal[PODCAST_SUPERVISOR_AGENT]]:
#     print("====PODCAST AUDIO GENERATOR NODE=====")
#     print("===STATE=====")
#     print(state)
#     previous_messages = state["messages"]
#     system_prompt = "Your job is to generate the audio file for the podcast."
#     invoke_message = [{"role": "system", "content": system_prompt},] + previous_messages
#     result = llm.invoke(invoke_message)
#     print("===RESULT=====")
#     print(result)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result.content, name=PODCAST_AUDIO_GENERATOR_AGENT)
#             ]
#         },
#         goto=PODCAST_SUPERVISOR_AGENT,
#     )


role_of_each_news_supervisor_worker = {
    TOP_HEADLINES_SUPERVISOR_AGENT: "Supervisor agent who is tasked with providing the top headlines along with the summary of the news.",
    PODCAST_SUPERVISOR_AGENT: "Supervisor agent who is tasked with providing the podcast script and save the transcript to a file.",
}

role_of_each_top_headlines_worker = {
    TOP_HEADLINES_AGENT: "Agent who is tasked with providing the top headlines with full content.",
    TOP_HEADLINES_SUMMARIZER_AGENT: "Agent who is tasked with summarizing the top headlines.",
}

role_of_each_podcast_worker = {
    PODCAST_TRANSCRIPT_GENERATOR_AGENT: "Script generator agent who is tasked with generating a podcast script for the top headlines and save the transcript to a file.",
}

# Tools
top_headlines_agent_tools = [get_top_headlines]
podcast_transcript_generator_agent_tools = [write_to_file]
# podcast_audio_generator_agent_tools = [create_podcast]

# LLM
llm = ChatOpenAI(model_name="gpt-4.1")


#NEWS SUPERVISOR
news_supervisor_members = list(role_of_each_news_supervisor_worker.keys())
news_supervisor_agent = make_news_supervisor_node(llm, members=news_supervisor_members, role_of_each_worker=role_of_each_news_supervisor_worker)

# TOP HEADLINES SUPERVISOR
top_headlines_members = list(role_of_each_top_headlines_worker.keys())
top_headlines_supervisor_agent = make_top_headlines_supervisor_node(llm, members=top_headlines_members, role_of_each_worker=role_of_each_top_headlines_worker)

# PODCAST SUPERVISOR
podcast_supervisor_members = list(role_of_each_podcast_worker.keys())
podcast_supervisor_agent = make_podcast_supervisor_node(llm, members=podcast_supervisor_members, role_of_each_worker=role_of_each_podcast_worker)