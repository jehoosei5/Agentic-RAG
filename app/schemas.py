from pydantic import BaseModel


class QuestionRequest(BaseModel):

    question: str

    session_id: str | None = None


class QuestionResponse(BaseModel):

    answer: str

    sources: list[str]

    retries: int