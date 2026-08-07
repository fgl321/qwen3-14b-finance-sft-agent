from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    thread_id: str | None = Field(default=None, max_length=100)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    request_id: str
    model: str
    answer: str
    usage: dict
