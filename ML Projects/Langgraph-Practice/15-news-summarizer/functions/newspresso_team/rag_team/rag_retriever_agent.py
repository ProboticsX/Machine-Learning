from common_imports import *
from constants import *
from classes import AgentState, RAGClass
from tools.helper_tools.tools import *
from functions.newspresso_team.rag_team.rag_supervisor_agent import role_of_each_rag_worker

def rag_retriever_node(state: AgentState) -> Command[Literal[RAG_SUPERVISOR_AGENT]]:
    print("====RAG RETRIEVER NODE=====")
    print("===STATE AT RAG RETRIEVER NODE=====")
    print(state)
    user_question = state["messages"][0].content
    system_prompt = f"{role_of_each_rag_worker[RAG_RETRIEVER_AGENT]}"+"""
    You are given a user question. You need to retrieve the top headlines from the pinecone database that are relevant to the user question. \n
    You need to return the top documents that are relevant to the user question. \n
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the user question: {user_question}"),
    ])
    formatted_prompt = prompt.format(user_question=user_question)
    rag_retriever_agent = create_react_agent(llm, 
                                             tools=rag_retriever_agent_tools, 
                                             prompt = formatted_prompt,
                                             response_format=RAGClass)
    invoke_message = {"user_question": user_question}
    result = rag_retriever_agent.invoke(invoke_message)
    print("===RESULT OF RAG RETRIEVER NODE=====")
    print(result)

    rag_class = RAGClass(
            date=result["structured_response"]["date"],
            rag_retriever_results=result["structured_response"]["rag_retriever_results"]
    )
    return Command(
        update={
            "messages": [
                HumanMessage(content="RAG retriever fetched the relevant headlines successfully.", name=RAG_RETRIEVER_AGENT)
            ],
            "rag_class": rag_class,
        },
        goto=RAG_SUPERVISOR_AGENT,
    )