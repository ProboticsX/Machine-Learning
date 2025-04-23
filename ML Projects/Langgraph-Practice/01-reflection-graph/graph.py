from common_imports import *
from functions import *

workflow = StateGraph(MessagesState)

workflow.add_node("generate", generate)  # generate
workflow.add_node("reflect", reflect)  # reflect

workflow.set_entry_point("generate")
workflow.add_conditional_edges(
    "generate",
    generate_router,
    {
        "reflect": "reflect",
        END: END
    },
)
workflow.add_edge("reflect", "generate")
graph = workflow.compile()