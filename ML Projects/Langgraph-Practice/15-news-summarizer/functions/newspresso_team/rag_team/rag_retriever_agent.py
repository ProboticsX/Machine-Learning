from common_imports import *
from constants import *
from classes import AgentState, RAGRetrieverClass, RAGClass
from tools.helper_tools.tools import *
from functions.newspresso_team.rag_team.rag_supervisor_agent import role_of_each_rag_worker

def rag_retriever_node(state: AgentState) -> Command[Literal[RAG_GRADER_AGENT]]:
    print("====RAG RETRIEVER NODE=====")
    print("===STATE AT RAG RETRIEVER NODE=====")
    print(state)
    user_question = state["user_question"]
    top_k = TOP_K_FOR_RAG_RETRIEVER
    system_prompt = f"{role_of_each_rag_worker[RAG_RETRIEVER_AGENT]}"+"""
    You are given a user question. You need to return the top k documents in output. \n
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the user question: {user_question} and the value of top_k: {top_k}"),
    ])

    formatted_prompt = prompt.format(user_question=user_question, top_k=top_k)
    rag_retriever_agent = create_react_agent(llm, 
                                             tools=rag_retriever_agent_tools, 
                                             prompt = formatted_prompt,
                                             response_format=RAGRetrieverClass)
    invoke_message = {"input": "Please do the task as per the system prompt"}
    result = rag_retriever_agent.invoke(invoke_message)
    print("===RESULT OF RAG RETRIEVER NODE=====")
    print(result)

    # Extract the tool call output from the result
    tool_output = None
    for message in result["messages"]:
        if isinstance(message, ToolMessage) and message.name == "retrieve_headlines_from_pinecone":
            tool_output = json.loads(message.content)  # Parse the JSON string into a Python dictionary
            break

    rag_retriever_class = RAGRetrieverClass(
            date=result["structured_response"]["date"],
            rag_retriever_results=tool_output
    )
    return Command(
        update={
            "messages": [
                HumanMessage(content="RAG retriever fetched the relevant headlines successfully.", name=RAG_RETRIEVER_AGENT)
            ],
            "rag_class": RAGClass(rag_retriever_class=rag_retriever_class),
        },
        goto=RAG_GRADER_AGENT,
    )

role_of_each_rag_retriever_worker = {
    RAG_GRADER_AGENT: "Agent who is tasked with grading the relevance of retrieved documents to a user question.",
}