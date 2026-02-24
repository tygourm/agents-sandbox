from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

agent = create_agent(GenericFakeChatModel(messages=iter([AIMessage("Hello, World!")])))
