from typing import Union, Any

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_gigachat import GigaChat
from pydantic import BaseModel, ConfigDict

type AgentResponse = AIMessage


class BaseAgent(BaseModel):
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Union[Runnable, GigaChat]
    prompt: str

    @property
    def _chain(self):
        model = self.model.with_retry()
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            ("placeholder", "{messages}")
        ])
        return prompt_template | model

    def invoke(self, query: dict) -> AgentResponse:
        try:
            response = self._chain.invoke(query)
            return response
        except Exception as e:
            raise e


class Agent(BaseAgent):
    binds: dict[str, str] = {}

    def bind(self, values: dict[str, str]):
        self.binds = {**self.binds, **values}

    def _send_messages(self, messages: list[AnyMessage]) -> AgentResponse:
        try:
            return self.invoke({**self.binds, "messages": messages})
        except Exception as e:
            raise e

    def send_messages(self, messages: list[AnyMessage]) -> AgentResponse:
        return self._send_messages(messages)

    def send_message(self, message: AnyMessage, messages: list[AnyMessage] = None) -> AgentResponse:
        return self._send_messages(messages + [message] if messages else [message])

    def send_human_message(self, content: str, messages: list[AnyMessage] = None) -> AgentResponse:
        return self.send_message(HumanMessage(content=content), messages)

    def send_ai_message(self, content: str, messages: list[AnyMessage] = None) -> AgentResponse:
        return self.send_message(AIMessage(content=content), messages)
