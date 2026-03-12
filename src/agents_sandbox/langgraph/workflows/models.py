from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def create_model() -> BaseChatModel:
    return ChatOpenAI(
        model="openai/gpt-oss-120b",  # ty: ignore[unknown-argument]
        base_url="https://albert.api.etalab.gouv.fr/v1",  # ty: ignore[unknown-argument]
    )
