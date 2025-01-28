from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from LangGraph.project_4.graph.chains.answer_grader import answer_grader
from LangGraph.project_4.graph.chains.hallucination_grader import hallucination_grader
from LangGraph.project_4.graph.consts import WEBSEARCH, GENERATE, RETRIEVE, GRADE_DOCUMENTS
from LangGraph.project_4.graph.nodes.generate import generate
from LangGraph.project_4.graph.nodes.grade_documents import grade_documents
from LangGraph.project_4.graph.nodes.retrieve import retrieve
from LangGraph.project_4.graph.nodes.web_search import web_search
from LangGraph.project_4.graph.state import GraphState

load_dotenv()
def decide_to_generate(state):
    print("--ASSESS GRADED DOCUMENTS--")
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def grade_generated_answer_grounded_in_documents_and_question(state: GraphState) -> str:
    print("--CHECK HALLUCINATIONS--")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    hallucination_score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if hallucination_grade := hallucination_score.binary_score:
        print("--DECISION: GENERATION IS GROUNDED IN DOCUMENTS--")
        print("--GRADE GENERATION vs QUESTION--")
        answer_score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_grade := answer_score.binary_score:
            print("--DECISION: GENERATION ADDRESSES QUESTION--")
            return "useful"
        else:
            print("--DECISION: GENERATION DOES NOT ADDRESS QUESTION--")
            return "not useful"
    else:
        print("--DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS--")
        return "not supported"



workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE
    }
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generated_answer_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH
    }
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="self_graph.png")