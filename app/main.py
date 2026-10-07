from fastapi import FastAPI

from app.schemas import (
    QuestionRequest,
    QuestionResponse,
)

from app.graph.graph import build_graph


app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0",
)


graph = build_graph()


@app.get("/")
def root():

    return {
        "message": "Agentic RAG API"
    }


@app.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: QuestionRequest
):

    initial_state = {
        "question": request.question,
        "rewritten_question": "",
        "source": "",
        "sources": [],
        "documents": [],
        "answer": "",
        "messages": [],
        "retry_count": 0,
        "document_relevant": False,
        "answer_grounded": False,
        "answers_question": False,
        "grading_feedback": "",
    }

    result = graph.invoke(
        initial_state
    )

    sources = list(
        {
            doc.metadata.get(
                "source",
                "unknown"
            )
            for doc in result.get(
                "documents",
                []
            )
        }
    )

    return QuestionResponse(
        answer=result["answer"],
        sources=sources,
        retries=result.get(
            "retry_count",
            0
        ),
    )