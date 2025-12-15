def calculate(expression: str) -> str:
    """
    Simple calculator tool.
    WARNING: uses eval for demo/academic purposes only.
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"
