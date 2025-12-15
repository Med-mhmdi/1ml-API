from agent_api.tools.calculator import calculate
from agent_api.tools.image_tool import classify_image
from llm_api.api import generate_text_internal


def agent_decide_and_execute(query: str, image_base64: str | None):
    """
    Core agent logic:
    - decide tool
    - execute tool
    - return final answer
    """

    # Decision logic (simple rules for academic clarity)
    if image_base64:
        tool = "image_classifier"
        result = classify_image(image_base64)
        return result, tool

    if any(char.isdigit() for char in query):
        tool = "calculator"
        result = calculate(query)
        return result, tool

    # Default: LLM
    tool = "llm"
    result = generate_text_internal(query)
    return result, tool
