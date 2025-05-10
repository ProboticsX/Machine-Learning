from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass

def displayGraph(graph):
    # print(graph.get_graph().draw_ascii())
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

def make_news_supervisor_node(llm, members) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def news_supervisor_node(state: AgentState) -> Command[Literal[*members, END]]:
        print("====NEWS SUPERVISOR NODE=====")
        print("===STATE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        invoke_message = [{"role": "system", "content": system_prompt},] + [last_message]
        print("===INVOKE MESSAGE=====")
        print(invoke_message)
        llm_with_structured_output = llm.with_structured_output(Router)
        response = llm_with_structured_output.invoke(invoke_message)
        print("===RESPONSE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return news_supervisor_node


def make_top_headlines_supervisor_node(llm, members) -> str:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor who is expected to provide the top news headlines along with the summary of the news."
        "You are tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
        " Do not perform any task yourself. Just route the request to any of the workers."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: Literal[*options]

    def top_headlines_supervisor_node(state: AgentState) -> Command[Literal[*members, NEWS_SUPERVISOR_AGENT]]:
        print("====TOP HEADLINES SUPERVISOR NODE=====")
        print("===STATE=====")
        print(state)
        """An LLM-based router."""
        last_message = state["messages"][-1]
        if state.get("top_headlines_summary") is not None:
            last_message = state["top_headlines_summary"]
        invoke_message = [{"role": "system", "content": system_prompt},] + [last_message]
        print("===INVOKE MESSAGE=====")
        print(invoke_message)
        llm_with_structured_output = llm.with_structured_output(Router)
        response = llm_with_structured_output.invoke(invoke_message)
        print("===RESPONSE=====")
        print(response)
        goto = response["next"]
        if goto == "FINISH":
            goto = NEWS_SUPERVISOR_AGENT
        print("===GOTO=====")
        print(goto)
        return Command(goto=goto, update={"next": goto})

    return top_headlines_supervisor_node

# def make_podcast_supervisor_node(llm, members) -> str:
#     options = ["FINISH"] + members
#     system_prompt = (
#         "You are a supervisor who is expected to provide the podcast script along with the audio file."
#         "You are tasked with managing a conversation between the"
#         f" following workers: {members}. Given the following user request,"
#         " respond with the worker to act next. Each worker will perform a"
#         " task and respond with their results and status. When finished,"
#         " respond with FINISH."
#         " Do not perform any task yourself. Just route the request to any of the workers."
#     )

#     class Router(TypedDict):
#         """Worker to route to next. If no workers needed, route to FINISH."""

#         next: Literal[*options]

#     def podcast_supervisor_node(state: AgentState) -> Command[Literal[*members, NEWS_SUPERVISOR_AGENT]]:
#         print("====PODCAST SUPERVISOR NODE=====")
#         print("===STATE=====")
#         print(state)
#         """An LLM-based router."""
#         last_message = state["messages"][-1]
#         invoke_message = [{"role": "system", "content": system_prompt},] + [last_message]
#         print("===INVOKE MESSAGE=====")
#         print(invoke_message)
#         llm_with_structured_output = llm.with_structured_output(Router)
#         response = llm_with_structured_output.invoke(invoke_message)
#         print("===RESPONSE=====")
#         print(response)
#         goto = response["next"]
#         if goto == "FINISH":
#             goto = NEWS_SUPERVISOR_AGENT
#         print("===GOTO=====")
#         print(goto)
#         return Command(goto=goto, update={"next": goto})




def top_headlines_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES NODE=====")
    print("===STATE=====")
    print(state)
    top_headlines_agent = create_react_agent(llm, 
                                             tools=top_headlines_agent_tools, 
                                             prompt = "Your job is to provide the top headlines using the tools provided.")
    result = top_headlines_agent.invoke(state)
    print("===RESULT=====")
    print(result)
    tool_message = result["messages"][-2]
    return Command(
        update={
            "messages": [
                HumanMessage(content=tool_message.content, name=TOP_HEADLINES_AGENT)
            ],
            "top_headlines": TopHeadlinesClass(
                top_headlines_full_content_from_tool=tool_message.content,
            ),
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )

def top_headlines_summarizer_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES SUMMARIZER NODE=====")
    print("===STATE=====")
    print(state)
    top_headlines_full_content_from_tool = state["top_headlines"]["top_headlines_full_content_from_tool"]
    system_prompt = """
                     Your job is to summarize the top headlines. \n
                     Remember to summarize each headline in about 200-300 words. \n
                     You will be given the top news headlines to you by the user.
                     """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the top headlines: {top_headlines_full_content_from_tool}"),
    ])
    top_headlines_summarizer_chain = prompt | llm
    invoke_message = {"top_headlines_full_content_from_tool": top_headlines_full_content_from_tool}
    result = top_headlines_summarizer_chain.invoke(invoke_message)
    print("===RESULT=====")
    print(result)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result.content, name=TOP_HEADLINES_SUMMARIZER_AGENT)
            ],
            "top_headlines": TopHeadlinesClass(
                top_headlines_summary=result.content
            ),
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )

# def podcast_transcript_generator_node(state: AgentState) -> Command[Literal[PODCAST_SUPERVISOR_AGENT]]:
#     print("====PODCAST TRANSCRIPT GENERATOR NODE=====")
#     print("===STATE=====")
#     print(state)
#     previous_messages = state["messages"]
#     system_prompt = """You will be given a summary of the top headlines. Your job is to generate a podcast script for the top headlines and save the transcript in a file called "podcast-script.txt".
#     The name of the podcast is "Newspresso".
#     The script should be in the following format:
#     <Person1> "Welcome to Newspresso – your personal generative AI podcast! We've got a jam-packed episode today covering everything from global politics to basketball buzzer-beaters. Let’s dive right in with a tense exchange at the White House."
#     </Person1><Person2> "Right—President Trump recently met with Canada’s new Prime Minister, Mark Carney, and let’s just say, things got frosty. Trump doubled down on his refusal to lower tariffs on Canadian imports, insisting they're justified. He even accused the U.S. of subsidizing Canada unfairly."
#     </Person2><Person1> "Yeah, and Carney tried to emphasize the economic interdependence between the two nations, but even he admitted a trade deal isn’t happening anytime soon. Those 25% tariffs are still on the table—and that’s a real strain, considering how closely the U.S. and Canada rely on one another."
#     </Person1><Person2> "Switching gears, let’s talk NBA playoffs. The Indiana Pacers pulled off a nail-biter against the Cleveland Cavaliers, winning 120 to 119!"
#     </Person2><Person1> "Oh, what a finish! Tyrese Haliburton hit a last-second three to seal the deal. Myles Turner and Aaron Nesmith both had huge nights with 23 points apiece, but man, Donovan Mitchell dropping 48 and still losing? That’s brutal."
#     </Person1><Person2> "The Pacers now lead the series 2-0, and all eyes are on Game 3 later this week. That’s going to be a must-watch."
#     </Person2>
#     """
#     podcast_transcript_generator_agent = create_react_agent(llm, 
#                                              tools=podcast_transcript_generator_agent_tools, 
#                                              prompt = system_prompt)
#     result = podcast_transcript_generator_agent.invoke(state)
#     print("===RESULT=====")
#     print(result)
#     return Command(
#         update={
#             "messages": [
#                 HumanMessage(content=result.content, name=PODCAST_TRANSCRIPT_GENERATOR_AGENT)
#             ]
#         },
#         goto=PODCAST_SUPERVISOR_AGENT,
#     )

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


# Tools
top_headlines_agent_tools = [get_top_headlines]
# podcast_transcript_generator_agent_tools = [write_to_file]

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini")


#NEWS SUPERVISOR
news_supervisor_members = [TOP_HEADLINES_SUPERVISOR_AGENT]
news_supervisor_agent = make_news_supervisor_node(llm, members=news_supervisor_members)

# TOP HEADLINES SUPERVISOR
top_headlines_members = [TOP_HEADLINES_AGENT, TOP_HEADLINES_SUMMARIZER_AGENT]
top_headlines_supervisor_agent = make_top_headlines_supervisor_node(llm, members=top_headlines_members)

# PODCAST SUPERVISOR
# podcast_supervisor_members = [PODCAST_TRANSCRIPT_GENERATOR_AGENT, PODCAST_AUDIO_GENERATOR_AGENT]
# podcast_supervisor_agent = make_podcast_supervisor_node(llm, members=podcast_supervisor_members)