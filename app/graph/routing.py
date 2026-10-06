def route_after_analysis(state):

    sources = state.get(
        "sources",
        []
    )

    if not sources:
        return "generate"

    if len(sources) == 1:
        return "retrieve"

    return "parallel_retrieve"

def route_after_document_grade(state):

    if state.get(
        "document_relevant",
        False
    ):
        return "generate"

    retry_count = state.get(
        "retry_count",
        0
    )

    if retry_count >= 3:
        return "fallback"

    return "rewrite"


def route_after_answer_grade(state):

    grounded = state.get(
        "answer_grounded",
        False
    )

    answers_question = state.get(
        "answers_question",
        False
    )

    if grounded and answers_question:
        return "end"

    retry_count = state.get(
        "retry_count",
        0
    )

    if retry_count >= 3:
        return "end"

    return "regenerate"

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