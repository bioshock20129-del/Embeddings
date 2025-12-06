import json
import random
from typing import Any, Callable

import numpy as np
from langchain_core.messages import HumanMessage, AIMessage
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

    def _filter_by_author(self, field: str, author: str):
        if len(self.messages) == 0:
            return []

        if field not in self.messages[0].keys():
            return []

        return list(map(lambda x: x.get(field), filter(lambda x: x.get("author") == author, self.messages)))

    def messages_by_author(self, author: str) -> list[str]:
        return self._filter_by_author("message", author)

    def scores_by_author(self, author: str) -> list[float]:
        return self._filter_by_author("score", author)

    def normalize_scores_by_author(self, author: str) -> list[float]:
        scores = self.scores_by_author(author)
        s_min, s_max = np.min(scores), np.max(scores)
        norm_scores = list(map(lambda x: (x - s_min) / (s_max - s_min), scores))

        return norm_scores

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
        bad_guy: Agent,
        start_message: str,
        size_dialogue: int,
        score_fn: Callable[[str, str], float],
) -> HistoryDialogue:
    history = HistoryDialogue(capacity=size_dialogue + 1).push("Dev", start_message)
    rnd = random.Random()

    def call_agent(agent, msg):
        messages = []
        for content in history.messages:
            if content["author"] == agent.name:
                messages.append(HumanMessage(content["message"]))
            else:
                messages.append(AIMessage(content["message"]))

        call_bad_guy = rnd.randint(0, 100)
        if call_bad_guy >= 50:
            is_call_bad_guy = True
            response = bad_guy.send_human_message(msg)
        else:
            is_call_bad_guy = False
            response = agent.send_human_message(msg)

        score = score_fn(agent.prompt, response.content)
        if is_call_bad_guy:
            score -= 0.8
            score = 0 if score < 0 else score

        history.push(agent.name, response.content, score)

    is_agent_1 = True
    while not history.is_full:
        print(f"{history.last_message["author"]}: {history.last_message["message"]}")
        current_agent = agent_1 if is_agent_1 else agent_2
        call_agent(current_agent, history.last_message["message"])
        is_agent_1 = not is_agent_1

    return history
