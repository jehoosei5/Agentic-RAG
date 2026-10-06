from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):

    needs_retrieval: bool = Field(
        description=(
            "Whether the question requires "
            "information from the documentation."
        )
    )

    sources: list[
        Literal[
            "fastapi",
            "langgraph",
            "sqlalchemy",
            "company"
        ]
    ] = Field(
        default_factory=list,
        description=(
            "Documentation sources needed "
            "to answer the question."
        )
    )


class DocumentGrade(BaseModel):

    relevant: bool = Field(
        description=(
            "Whether the retrieved documents "
            "are relevant to the question."
        )
    )

    reasoning: str = Field(
        description="Explain why the documents are or are not relevant."
    )


class AnswerGrade(BaseModel):

    grounded: bool = Field(
        description=(
            "Whether the answer is supported "
            "by the retrieved documents."
        )
    )

    answers_question: bool = Field(
        description=(
            "Whether the answer directly "
            "answers the user's question."
        )
    )

    feedback: str = Field(
        description="Explain any problems with the answer."
    )


class RewriteQuery(BaseModel):

    rewritten_question: str = Field(
        description=(
            "A clearer and more specific search query "
            "for the documentation."
        )
    )