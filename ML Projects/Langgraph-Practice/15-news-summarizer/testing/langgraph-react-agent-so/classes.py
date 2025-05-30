from pydantic import Field
from typing_extensions import TypedDict

class BooleanClass(TypedDict):
    boolean: bool = Field(description="The boolean value")