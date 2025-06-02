from common_imports import *
from constants import *
from classes import AgentState, RAGGeneratorClass
from tools.helper_tools.tools import *
from functions.newspresso_team.rag_team.rag_grader_agent import role_of_each_rag_grader_worker

def rag_generator_node(state: AgentState) -> Command[Literal[RAG_SUPERVISOR_AGENT]]:
    print("====RAG GENERATOR NODE=====")
    print("===STATE AT RAG GENERATOR NODE=====")
    print(state)
    user_question = state["user_question"]
    filtered_documents = state["rag_class"]["rag_grader_class"]["filtered_documents"]
    rag_grader_final_result = state["rag_class"]["rag_grader_class"]["rag_grader_final_result"]
    system_prompt = f"{role_of_each_rag_grader_worker[RAG_GENERATOR_AGENT]}"+"""
    If you can't answer the question, you need to say that you can't answer the question. \n
    If the question is answerable, you need to answer the question in an elaborate manner. \n
    You can also use the documents attached to enrich your answer in case it's an answerable question. \n
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the user question: {user_question}. (Yes/No) whether the question is answerable or not: {rag_grader_final_result}. The filtered documents (in case the question is answerable): {filtered_documents}"),
    ])
    invoke_message = {"input": "Please do the task as per the system prompt"}
    formatted_prompt = prompt.format(user_question=user_question, rag_grader_final_result=rag_grader_final_result, filtered_documents=filtered_documents)
    rag_generator_agent = create_react_agent(llm, 
                                             tools=[],
                                             prompt = formatted_prompt,
                                             response_format=RAGGeneratorClass
                                             )
    result = rag_generator_agent.invoke(invoke_message)
    print("===RESULT OF RAG GENERATOR NODE=====")
    print(result)

    rag_generator_class = RAGGeneratorClass(
            rag_generator_final_answer=result["structured_response"]["rag_generator_final_answer"],
            sources=result["structured_response"]["sources"]
    )
    if state.get("rag_class") is not None:
        rag_class = state["rag_class"].copy()
        rag_class["rag_generator_class"] = rag_generator_class

    return Command(
        update={
            "messages": [
                HumanMessage(content="RAG generator generated the final answer successfully.", name=RAG_GENERATOR_AGENT)
            ],
            "rag_class": rag_class,
        },
        goto=RAG_SUPERVISOR_AGENT,
    )