from common_imports import *
from classes import AgentDetails

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="basic_rag_graph.png")

def agent(state):
    print("=====AGENT======")
    print(state)
    question = state["question"]
    system_prompt = """You are an agent whose job is to judge whether the user's question is related to US politics.\n
    If it is, you should return "yes". Otherwise, you should return "no".\n
    You will be given the question and you should adhere to the above instructions.\n """
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {question}"),
    ])
    agent_chain = agent_prompt | llm_with_structured_output_agent
    invoke_message = {"question": question}
    print("========INVOKE MESSAGE========")
    print(invoke_message)
    result = agent_chain.invoke(invoke_message)
    print("========RESULT========")
    print(result)
    return {"agent_details": result}

def llm_function(state):
    print("=====LLM FUNCTION======")
    print(state)
    question = state["question"]
    system_prompt = """You need to answer the user's question based on your knowledge"""
    llm_function_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {question}"),
    ])
    llm_function_chain = llm_function_prompt | llm
    invoke_message = {"question": question}
    print("========INVOKE MESSAGE========")
    print(invoke_message)
    result = llm_function_chain.invoke(invoke_message)
    print("========RESULT========")
    print(result)
    return {"messages": [result]}

def rag(state):
    print("=====RAG======")
    print(state)
    question = state["question"]
    context = retriever.invoke(question)
    system_prompt = """You are a helpful assistant that can answer questions about the user's question based on your knowledge.\n
                       You will be given the question and the knowledge base as context.\n """
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {question}, Context: {context}"),
    ])
    rag_chain = rag_prompt | llm
    invoke_message = {"question": question, "context": context}
    print("========INVOKE MESSAGE========")
    print(invoke_message)
    result = rag_chain.invoke(invoke_message)
    print("========RESULT========")
    print(result)
    return {"messages": [result]}

def agent_router(state):
    print("=====AGENT ROUTER======")
    print(state)
    binary_score = state["agent_details"].binary_score
    if binary_score == "yes":
        return RAG
    else:
        return LLM

embeddings = OpenAIEmbeddings()

documents = TextLoader("../data/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, embeddings).as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model_name="gpt-4o-mini")
llm_with_structured_output_agent = llm.with_structured_output(AgentDetails)