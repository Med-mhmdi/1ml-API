AGENT_METRICS = {
    "total_requests": 0,
    "calculator_calls": 0,
    "image_calls": 0,
    "llm_calls": 0,
}


def record_tool(tool_name: str):
    AGENT_METRICS["total_requests"] += 1

    if tool_name == "calculator":
        AGENT_METRICS["calculator_calls"] += 1
    elif tool_name == "image_classifier":
        AGENT_METRICS["image_calls"] += 1
    elif tool_name == "llm":
        AGENT_METRICS["llm_calls"] += 1
