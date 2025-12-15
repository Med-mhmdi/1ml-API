from fastapi import APIRouter
from agent_api.schemas import AgentRequest, AgentResponse
from agent_api.core import agent_decide_and_execute

router = APIRouter(
    prefix="/agent",
    tags=["AI Agent"]
)


@router.post("/ask", response_model=AgentResponse)
def ask_agent(req: AgentRequest):
    answer, tool_used = agent_decide_and_execute(
        query=req.query,
        image_base64=req.image_base64
    )

    return AgentResponse(
        answer=answer,
        tool_used=tool_used
    )
