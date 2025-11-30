import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, AIMessage
from langchain_gigachat import GigaChatEmbeddings
from pydantic import BaseModel

from src.llm.agent import Agent
from src.utils import save_to_file


class HistoryDialogue(BaseModel):
    messages: list[dict[str, Any]] = []
    capacity: int = 0

    @property
    def is_full(self) -> bool:
        return len(self.messages) == self.capacity

    @property
    def last_message(self):
        return self.messages[-1]

    def push(self, author: str, message: str, score: float = 0):
        self.messages.append({"author": author, "message": message, "score": score})
        return self

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4, ensure_ascii=False)

    def save_to_file(self, path: str):
        save_to_file(lambda f: f.write(self.to_json()), path)


def read_from_file(path: str) -> HistoryDialogue:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
        return HistoryDialogue(
            messages=content["messages"],
            capacity=content["capacity"],
        )


def generate_from_llm(
        agent_1: Agent,
        agent_2: Agent,
        start_message: str,
        size_dialogue: int,
        score_fn: Callable[[str, str], float],
) -> HistoryDialogue:
    history = HistoryDialogue(capacity=size_dialogue).push("Dev", start_message)

    def call_agent(agent, msg):
        messages = []
        for content in history.messages:
            if content["author"] == agent.name:
                messages.append(HumanMessage(content["message"]))
            else:
                messages.append(AIMessage(content["message"]))

        response = agent.send_human_message(msg, messages)
        score = score_fn(agent.prompt, response.content)
        history.push(agent.name, response.content, score)

    is_agent_1 = True
    while not history.is_full:
        print(f"{history.last_message["author"]}: {history.last_message["message"]}")
        current_agent = agent_1 if is_agent_1 else agent_2
        call_agent(current_agent, history.last_message["message"])
        is_agent_1 = not is_agent_1

    return history
