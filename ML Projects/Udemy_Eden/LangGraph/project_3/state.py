import operator
from typing import TypedDict, Union, Annotated

from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] #in order to add the intermediate step in between to the agent