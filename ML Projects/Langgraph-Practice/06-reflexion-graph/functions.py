from common_imports import *
from classes import DraftDetails, ExecuteToolDetails, RevisorDetails

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="reflexion_graph.png")

def draft(state):
    print("========DRAFT========")
    print(state)
    topic = state['topic']
    first_instruction = state['first_instruction']
    system_prompt = """You are expert researcher. You need to research on a given topic by following the instructions below: \n
                1. Find the first instruction to follow below.\n
                2. Reflect and critique your answer. Be severe to maximize improvement.\n
                3. Separately, recommend 1-3 search queries to research information and improve your answer. These should not overlap with the reflection."""
    draft_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here's the given topic to research: {topic}\n Here's the first instruction: {first_instruction}"),
    ])
    draft_chain = draft_prompt | llm_with_structured_output_draft
    invoke_message = {"topic": topic, "first_instruction": first_instruction}
    print("========INVOKE MESSAGE========")
    print(invoke_message)
    result = draft_chain.invoke(invoke_message)
    print("========RESULT========")
    print(result)
    return {"draft_details": result}

def execute_tools(state):
    print("========EXECUTE TOOLS========")
    print(state)
    system_prompt = """You are given a list of search queries to research information and improve your answer. You need to use the tools provided to return the results based on the search queries."""
    search_queries = state['draft_details'].search_queries
    if state.get("revisor_details", None) is not None:
        search_queries = state['revisor_details'].search_queries
    execute_tool_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here's the list of search queries: {search_queries}"),
    ])
    execute_tool_chain = execute_tool_prompt | llm_with_structured_output_execute_tool
    invoke_message = {"search_queries": search_queries}
    print("========INVOKE MESSAGE========")
    print(invoke_message)
    result = execute_tool_chain.invoke(search_queries)
    print("========RESULT========")
    print(result)
    return {"execute_tool_details": result}

def revisor(state):
    print("========REVISOR========")
    print(state)
    topic = state['topic']
    first_instruction = state['first_instruction']
    context = state['execute_tool_details']
    system_prompt = """Revise your previous answer using the new information.\n
    - You should rewrite the summary to use the previous critique to add important information to your answer from what was missing before. Moreover, you should use the previous critique to remove superfluous information from your answer.\n
    - You should rewrite the missing information in the newly rewritten summary.\n
    - You should also rewrite the superfluous information in the newly rewritten summary.\n
    - You should also rewrite the search queries to research information and improve your answer.\n
    - You MUST include numerical citations in your revised answer to ensure it can be verified. \n
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of: \n
            - [1] https://example.com \n
            - [2] https://example.com \n
    - You will also be given pieces of information by the user:
        - Original topic
        - Original first instruction
        - Previous answer including the previous summary, missing information, superfluous information and search queries
"""
    revisor_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here's the original topic: {topic}\n and original first instruction: {first_instruction}\n Here's the previous answer: {context}"),
    ])
    revisor_chain = revisor_prompt | llm_with_structured_output_revisor
    invoke_message = {"context": context, "first_instruction": first_instruction, "topic": topic}
    print("========INVOKE MESSAGE========")
    print(invoke_message)
    result = revisor_chain.invoke(invoke_message)
    print("========RESULT========")
    print(result)
    return {"revisor_details": result}

def revisor_router(state):
    print("========REVISOR ROUTER========")
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return REVISOR

def get_stock_price(ticker: str) -> float:
    """Gets a stock price from Yahoo Finance.
    Args:
        ticker: ticker str
    """
    stock = yf.Ticker(ticker)
    return stock.info['previousClose']

def tavily_search(search_query: str):
    """Search the web for the query."""
    return search.invoke(search_query)

def tavily_search_list(search_queries: list[str]):
    """Search the web for the queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])

def get_tools():
    return [tavily_search, get_stock_price, tavily_search_list]

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=3)
llm = ChatOpenAI(model_name="gpt-4o-mini")
llm_with_tools = llm.bind_tools(get_tools())
llm_with_structured_output_draft = llm.with_structured_output(DraftDetails)
llm_with_structured_output_execute_tool = llm_with_tools.with_structured_output(ExecuteToolDetails)
llm_with_structured_output_revisor = llm.with_structured_output(RevisorDetails)

# temp_search_queries = ["Who is the president of the United States?", "What is the capital of Germany?", "What is the capital of Italy?"]
# temp_search_results = tavily_search_list(temp_search_queries)
# for result in temp_search_results:
#     print(result)
#     print("================")
# format of each result (per query):
# [{
#     "title": "Title of the search result",
#     "url": "URL of the search result",
#     "content": "Content of the search result",
#     "source": "Source of the search result",
# },
# {
#     "title": "Title of the search result",
#     "url": "URL of the search result",
#     "content": "Content of the search result",
#     "source": "Source of the search result",
# },]
