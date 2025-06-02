from common_imports import *
from constants import *
from classes import AgentState, RAGClass, RAGGraderClass
from tools.helper_tools.tools import *
from functions.newspresso_team.rag_team.rag_retriever_agent import role_of_each_rag_retriever_worker

def rag_grader_node(state: AgentState) -> Command[Literal[RAG_GENERATOR_AGENT]]:
    print("====RAG GRADER NODE=====")
    print("===STATE AT RAG GRADER NODE=====")
    print(state)
    user_question = state["user_question"]
    documents = state["rag_class"]["rag_retriever_class"]["rag_retriever_results"]
    date = state["rag_class"]["rag_retriever_class"]["date"]
    system_prompt = f"{role_of_each_rag_retriever_worker[RAG_GRADER_AGENT]}"+"""
    You are a helpful assistant that grades the relevance of documents to a question. \n
    You will be given a question and a document. \n
    The output should be a binary score of yes or no. \n
    If the document is relevant to the question, the output should be yes. \n
    If the document is not relevant to the question or there are no documents retrieved or the document is not from today's date, the output should be no. \n
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Here is the user question: {user_question} and the date: {date}. The document to be graded: {document} and the urls which are relevant to the document: {urls}"),
    ])
    invoke_message = {"input": "Please do the task as per the system prompt"}
    filtered_docs = []
    for document in documents:
        formatted_prompt = prompt.format(user_question=user_question, date=date, document=document['metadata']['content_summary'], urls=document['metadata']['sources'])
        rag_grader_agent = create_react_agent(llm, 
                                                tools=rag_grader_agent_tools,
                                                prompt = formatted_prompt
                                                )
        document_result = rag_grader_agent.invoke(invoke_message)
        print("===HERE IS THE RESULT OF THE DOCUMENT:", document['id'],"===")
        print(document_result)
        if document_result["messages"][-1].content.lower() == "yes":
            print("=====GRADER: DOCUMENT IS RELEVANT======")
            filtered_docs.append(document)
        else:
            print("=====GRADER: DOCUMENT IS NOT RELEVANT======")
    rag_grader_final_result = "yes"
    if len(filtered_docs) == 0:
        rag_grader_final_result = "no"
    print("===RESULT OF RAG GRADER NODE=====")
    print("RAG GRADER FINAL RESULT: ", rag_grader_final_result)
    print("FILTERED DOCUMENTS: ", filtered_docs)

    rag_grader_class = RAGGraderClass(
            rag_grader_final_result=rag_grader_final_result,
            filtered_documents=filtered_docs
    )
    if state.get("rag_class") is not None:
        rag_class = state["rag_class"].copy()
        rag_class["rag_grader_class"] = rag_grader_class

    if rag_grader_final_result == "no":
        return Command(
            update={
                "messages": [
                    HumanMessage(content="RAG grader graded the relevance of the documents successfully and no documents were found relevant to the user question. This question is not answerable.", name=RAG_GRADER_AGENT)
                ],
                "rag_class": rag_class,
            },
            goto=RAG_GENERATOR_AGENT,
        )
    
    return Command(
        update={
            "messages": [
                HumanMessage(content="RAG grader graded the relevance of the documents successfully. The documents are relevant to the user question. This question is answerable.", name=RAG_GRADER_AGENT)
            ],
            "rag_class": rag_class,
        },
        goto=RAG_GENERATOR_AGENT,
    )

role_of_each_rag_grader_worker = {
    RAG_GENERATOR_AGENT: "You are a helpful assistant that generates the final answer to the user question. \n"
}