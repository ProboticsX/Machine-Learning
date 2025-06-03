from elevenlabs.conversational_ai.conversation import ClientTools

def answer_user_question(user_question: str) -> str:
    return "This is a test answer"

client_tools = ClientTools()
client_tools.register(answer_user_question)