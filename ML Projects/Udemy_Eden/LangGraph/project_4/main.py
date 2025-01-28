from dotenv import load_dotenv

from LangGraph.project_4.graph.graph import app

load_dotenv()
if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "What is Agent Memory?"}))
    