from common_imports import *
from classes import GraderDetails, GeneratorRouterDetails, GroundedChainDetails, QuestionRouterDetails

def displayGraph(graph):
    # print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="adaptive_rag_graph.png")

def retriever_function(state):
    print("=====RETRIEVER FUNCTION======")
    print(state)
    question = state["question"]
    documents = retriever.invoke(question)
    return {"retriever_details": {"documents": documents}}

def grader_function(state):
    print("=====GRADER FUNCTION======")
    print(state)
    question = state["question"]
    documents = state["retriever_details"]["documents"]
    filtered_docs = []
    system_message = """
    You are a helpful assistant that grades the relevance of documents to a question. \n
    You will be given a question and a document. \n
    You need to grade the relevance of the document to the question. \n
    The output should be a binary score of yes or no. \n
    If the document is relevant to the question, the output should be yes. \n
    If the document is not relevant to the question, the output should be no. \n
    """
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "Question: {question} \n Document to be graded: {document}"),
    ])
    grader_chain = grader_prompt | llm_with_structured_output_grader
    web_search = False
    for doc in documents:
        document_result = grader_chain.invoke({"question": question, "document": doc.page_content})
        if document_result.binary_score == "yes":
            print("=====GRADER: DOCUMENT IS RELEVANT======")
            filtered_docs.append(doc)
        else:
            print("=====GRADER: DOCUMENT IS NOT RELEVANT======")
            web_search = True
    print("====RESULT FROM GRADER FUNCTION=====")
    print("Filtered Docs: ", filtered_docs)
    print("Web Search: ", web_search)
    return {"filtered_docs": filtered_docs, "web_search": web_search}


def websearch_function(state):
    print("=====WEBSEARCH FUNCTION======")
    print(state)
    question = state["question"]
    documents = []
    if state.get("filtered_docs", None) is not None:
        documents = state["filtered_docs"]
    web_search_tool = TavilySearchResults(max_results=3)
    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_results)
    print("====WEB RESULTS THROUGH TAVILY=====")
    print(web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    print("====RESULT FROM WEBSEARCH FUNCTION=====")
    print("Documents: ", documents)
    return {"filtered_docs": documents}

def generator_function(state):
    print("=====GENERATOR FUNCTION======")
    print(state)
    question = state["question"]
    documents = []
    if state.get("filtered_docs", None) is not None:
        documents = state["filtered_docs"]
    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = rag_prompt | llm
    invoke_message = {"question": question, "context": documents}
    print("=====INVOKE MESSAGE======")
    print(invoke_message)
    result = rag_chain.invoke(invoke_message)
    print("=====RESULT FROM GENERATOR FUNCTION======")
    print(result)
    return {"messages": [result]}

def transform_question_function(state):
    print("=====TRANSFORM QUESTION FUNCTION======")
    print(state)
    question = state["question"]
    system_prompt = """
    You are a helpful assistant that transforms the question to a more specific and detailed question.
    The output should be a question that is more specific and detailed.
    """
    transoform_question_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {question}"),
    ])
    input_message = {"question": question}
    transform_chain = transoform_question_prompt | llm
    result = transform_chain.invoke(input_message)
    print("=====RESULT FROM TRANSFORM QUESTION FUNCTION======")
    print(result)
    return {"question": result.content}

def grader_router(state):
    print("=====GRADER ROUTER======")
    print(state)
    web_search = state["web_search"]
    if web_search:
        return TRANSFORM_QUESTION
    else:
        return GENERATOR

def generator_router(state):
    print("=====GENERATOR ROUTER======")
    print(state)
    answer = state["messages"][-1].content
    question = state["question"]
    documents = state["filtered_docs"]
    system_prompt = """
    You are a helpful assistant that checks if the generated answer is grounded in the provided documents. \n
    The output should be a binary score of yes or no. \n
    If the answer is grounded in the documents, the output should be yes. \n
    If the answer is not grounded in the documents, the output should be no. \n
    """
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Generated Answer: {answer} \n Documents: {documents}"),
    ])
    generator_router_chain = generator_prompt | llm_with_structured_output_generator_router
    invoke_message = {"answer": answer, "documents": documents}
    print("=====INVOKE MESSAGE======")
    print(invoke_message)
    result = generator_router_chain.invoke(invoke_message)
    print("=====RESULT FROM GENERATOR ROUTER FUNCTION (GENERATION VS DOCUMENT GROUNDED)======")
    print(result)
    if result.binary_score == "yes":
        grounded_system_prompt = """
        You are a helpful assistant that checks if the generated answer is correctly answers the question or not. \n
        The output should be a binary score of yes or no. \n
        If the answer is correctly answers the question, the output should be yes. \n
        If the answer is not correctly answers the question, the output should be no. \n
    """
        grounded_prompt = ChatPromptTemplate.from_messages([
            ("system", grounded_system_prompt),
            ("user", "Generated Answer: {answer} \n Question: {question}"),
        ])
        grounded_chain = grounded_prompt | llm_with_structured_output_grounded_chain
        invoke_message = {"answer": answer, "question": question}
        print("=====INVOKE MESSAGE======")
        print(invoke_message)
        grounded_result = grounded_chain.invoke(invoke_message)
        print("=====RESULT FROM GROUNDED CHAIN (GENERATION VS QUESTION GROUNDED)======")
        print(grounded_result)
        if grounded_result.binary_score == "yes":
            return ANSWER_GROUNDED_AND_RELEVANT
        else:
            return ANSWER_NOT_RELEVANT
    else:
        return ANSWER_NOT_GROUNDED

def question_router(state):
    print("=====QUESTION ROUTER======")
    print(state)
    question = state["question"]
    system_prompt = """You are an expert at routing a user question to a vectorstore or web search.
            The vectorstore contains documents related to the US politics.
            Use the vectorstore for questions on these topics. For all else, use web-search."""
    question_router_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Question: {question}"),
    ])
    question_router_chain = question_router_prompt | llm_with_structured_output_question_router
    invoke_message = {"question": question}
    print("=====INVOKE MESSAGE======")
    print(invoke_message)
    result = question_router_chain.invoke(invoke_message)
    print("=====RESULT FROM QUESTION ROUTER FUNCTION======")
    print(result)
    if result.binary_score == "yes":
        return RETRIEVER
    else:
        return WEBSEARCH

embeddings = OpenAIEmbeddings()
documents = TextLoader("../data/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, embeddings).as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model_name="gpt-4o-mini")
llm_with_structured_output_grader = llm.with_structured_output(GraderDetails)
llm_with_structured_output_generator_router = llm.with_structured_output(GeneratorRouterDetails)
llm_with_structured_output_grounded_chain = llm.with_structured_output(GroundedChainDetails)
llm_with_structured_output_question_router = llm.with_structured_output(QuestionRouterDetails)