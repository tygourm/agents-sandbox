from langchain_core.tools import tool


@tool(parse_docstring=True)
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of the two numbers.

    """
    return a + b


@tool(parse_docstring=True)
def sub(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The difference of the two numbers.

    """
    return a - b
