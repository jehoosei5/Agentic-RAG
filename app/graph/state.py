from typing import Annotated
import operator

from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    question: str

    rewritten_question: str

    source: str

    sources: list[str]

    documents: Annotated[list[Document], operator.add]

    answer: str

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

    retry_count: int

    document_relevant: bool

    answer_grounded: bool

    answers_question: bool

    grading_feedback: str