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


llm = CHAT_MODEL

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

def retrieve_worker(state):

    question = state["question"]

    source = state["source"]

    documents = retrieve_documents(
        question=question,
        source=source,
    )

    return {
        "documents": documents
    }


def grade_documents(state):

    question = state["question"]

    documents = state.get(
        "documents",
        []
    )

    if not documents:

        return {
            "document_relevant": False
        }

    document_text = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    structured_llm = llm.with_structured_output(
        DocumentGrade
    )

    prompt = f"""
You are a document relevance grader.

Determine whether the retrieved documentation
contains useful information for answering the question.

Question:

{question}

Retrieved documentation:

{document_text}

Return relevant=true only if the documents
contain information that can meaningfully
help answer the question.
"""

    result = structured_llm.invoke(
        prompt
    )

    return {
        "document_relevant": result.relevant
    }

def rewrite_query(state):

    question = state["question"]

    structured_llm = llm.with_structured_output(
        RewriteQuery
    )

    prompt = f"""
You are a search query optimization agent.

The documentation retrieved for the user's
question was not relevant enough.

Rewrite the question into a more precise
technical search query.

Original question:

{question}

Return only the improved search query.
"""

    result = structured_llm.invoke(
        prompt
    )

    return {
        "question": result.rewritten_question,
        "rewritten_question": (
            result.rewritten_question
        ),
        "retry_count": (
            state.get("retry_count", 0) + 1
        ),
    }


def generate_answer(state):

    question = state["question"]

    documents = state.get(
        "documents",
        []
    )

    if documents:

        context = "\n\n".join(
            f"""
SOURCE: {doc.metadata.get("source")}
DOCUMENT: {doc.metadata.get("document")}

{doc.page_content}
"""
            for doc in documents
        )

    else:
        context = "No documentation was retrieved."

    prompt = f"""
You are a technical documentation assistant.

Answer the user's question.

If documentation is provided, use it as the
primary source of truth.

Do not invent facts that are not supported
by the documentation.

If the documentation does not contain enough
information, clearly say so.

Question:

{question}

Documentation:

{context}
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content
    }

def grade_answer(state):

    question = state["question"]

    answer = state["answer"]

    documents = state.get(
        "documents",
        []
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    structured_llm = llm.with_structured_output(
        AnswerGrade
    )

    prompt = f"""
You are an answer quality evaluator.

Evaluate the generated answer.

Question:

{question}

Retrieved documentation:

{context}

Generated answer:

{answer}

Determine:

1. Is the answer grounded in the documentation?
2. Does it actually answer the question?
3. What should be improved?
"""

    result = structured_llm.invoke(
        prompt
    )

    return {
        "answer_grounded": result.grounded,
        "answers_question": result.answers_question,
        "grading_feedback": result.feedback,
    }


def regenerate_answer(state):

    feedback = state.get(
        "grading_feedback",
        ""
    )

    question = state["question"]

    documents = state.get(
        "documents",
        []
    )

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
Improve the previous answer using the grader feedback.

Question:

{question}

Documentation:

{context}

Grader feedback:

{feedback}

Generate a new answer that:

- directly answers the question
- stays grounded in the documentation
- avoids unsupported claims
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content,
        "retry_count": (
            state.get("retry_count", 0) + 1
        ),
    }


def fallback(state):

    return {
        "answer": (
            "I could not find enough relevant "
            "information in the available documentation "
            "to answer this question reliably."
        )
    }