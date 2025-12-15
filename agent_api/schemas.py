from pydantic import BaseModel
from typing import Optional


class AgentRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None


class AgentResponse(BaseModel):
    answer: str
    tool_used: str
