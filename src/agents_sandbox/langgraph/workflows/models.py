from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

chat = ChatOpenAI(
    model="openai/gpt-oss-120b",  # ty: ignore[unknown-argument]
    base_url="https://albert.api.etalab.gouv.fr/v1",  # ty: ignore[unknown-argument]
)
