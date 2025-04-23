from common_imports import *

def get_llm():
    return ChatOpenAI()

def get_draft_chain(state):
    topic = state["messages"][0].content
    system_message = """You are an expert researcher on a given topic.
    1. You will be given a set of instructions to be followed.
    2. Reflect and critique your answer. Be severe to maximize improvement.
    3. Separately, recommend 1-3 search queries to research information and improve your answer. These should not overlap with the reflection."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "Here's the given topic: {topic} and the instructions: {instructions}"),
        ]
    )
    draft_chain = prompt | get_llm()
    return draft_chain, topic

def get_reflect_chain(state):
    system_message = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""