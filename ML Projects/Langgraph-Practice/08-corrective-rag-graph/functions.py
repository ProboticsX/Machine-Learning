from common_imports import *
from classes import GraderDetails

def displayGraph(graph):
    print(graph.get_graph().draw_ascii())
    graph.get_graph().draw_mermaid_png(output_file_path="corrective_rag_graph.png")

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
    if state["filtered_docs"] is not None:
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
    if state["filtered_docs"] is not None:
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

embeddings = OpenAIEmbeddings()
documents = TextLoader("../data/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, embeddings).as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model_name="gpt-4o")
llm_with_structured_output_grader = llm.with_structured_output(GraderDetails)