from common_imports import *
from constants import *
from classes import AgentState, TopHeadlinesClass, TopicClass
from tools.helper_tools.tools import *
from functions.newspresso_team.top_headlines_team.top_headlines_supervisor_agent import role_of_each_top_headlines_worker

def top_headlines_node(state: AgentState) -> Command[Literal[TOP_HEADLINES_SUPERVISOR_AGENT]]:
    print("====TOP HEADLINES NODE=====")
    print("===STATE AT TOP HEADLINES NODE=====")
    print(state)

    def get_topic_from_user_question(question: str) -> str:
        """Get the topic from the user request."""
        system_prompt = """You are given a user question and you need to extract the topic from it.
                       These are the topics you can choose from: \n
                        - business \n
                        - technology \n
                        - science \n
                        - health \n
                        - entertainment \n
                        - sports \n
                        - general (choose this if cannot find a specific topic) \n
                       """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Here's the user question: {question}"),
        ])
        llm_with_structured_output = llm.with_structured_output(TopicClass)
        topic_extractor_chain = prompt | llm_with_structured_output
        invoke_message = {"question": question}
        result = topic_extractor_chain.invoke(invoke_message)
        print("===RESULT OF TOPIC EXTRACTOR CHAIN=====")
        print(result)
        return result["topic"]

    user_question = state["messages"][0].content
    topic = get_topic_from_user_question(user_question)

    # Get the top headlines from the news API
    system_prompt = f"{role_of_each_top_headlines_worker[TOP_HEADLINES_AGENT]}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please fetch the top headlines for the topic: {topic}"),
    ])
    formatted_prompt = prompt.format(topic=topic)
    top_headlines_agent = create_react_agent(llm, 
                                             tools=top_headlines_agent_tools, 
                                             prompt = formatted_prompt)
    invoke_message = {"input": "Please fetch the top headlines"}
    result = top_headlines_agent.invoke(invoke_message)
    print("===RESULT OF TOP HEADLINES NODE=====")
    print(result)
    current_top_headlines = TopHeadlinesClass(
        top_headlines_processed_news_file=processsed_news_file_path
    )
    if state.get("top_headlines") is not None:
        current_top_headlines = state["top_headlines"].copy()
        current_top_headlines["top_headlines_processed_news_file"] = processsed_news_file_path
    return Command(
        update={
            "messages": [
                HumanMessage(content="Top headlines fetched successfully.", name=TOP_HEADLINES_AGENT)
            ],
            "top_headlines": current_top_headlines,
            "topic": TopicClass(topic=topic),
        },
        goto=TOP_HEADLINES_SUPERVISOR_AGENT,
    )