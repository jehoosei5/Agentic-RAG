from langchain_openai import AzureChatOpenAI

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from app.config import CHAT_MODEL
from app.graph.schemas import (
    RouteDecision,
    DocumentGrade,
    AnswerGrade,
    RewriteQuery,
)

from app.retrieval.retrievers import (
    retrieve_documents,
)


llm = AzureChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
)

def analyze_query(state):

    question = state["question"]

    structured_llm = llm.with_structured_output(
        RouteDecision
    )

    prompt = f"""
You are the routing agent for an Agentic RAG system.

Analyze the user's question.

Determine whether the question requires
information from the company's documentation.

Available documentation sources:

- fastapi
- langgraph
- sqlalchemy
- company

Select every source that could contain information
needed to answer the question.

If the question can be answered without documentation,
set needs_retrieval to false and return an empty sources list.

User question:

{question}
"""

    result = structured_llm.invoke(
        [
            SystemMessage(
                content=prompt
            ),
            HumanMessage(
                content=question
            ),
        ]
    )

    return {
        "source": (
            result.sources[0]
            if result.sources
            else "none"
        ),
        "sources": result.sources,
    }

def retrieve(state):

    question = state["question"]

    source = state["source"]

    documents = retrieve_documents(
        question=question,
        source=source,
    )

    return {
        "documents": documents
    }