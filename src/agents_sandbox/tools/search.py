from langchain_core.tools import tool
from langchain_tavily import TavilySearch


@tool(parse_docstring=True)
def search(query: str) -> dict:
    """Search the web for a query.

    Args:
        query: The query to search for.

    Returns:
        A dictionary containing the search results.

    """
    return TavilySearch().invoke(query)
