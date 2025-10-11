import json
import logging
import os

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_gigachat import GigaChat
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agent import Agent
from embeddings import embedding

# загружаем переменные окружения из .env
load_dotenv(find_dotenv())


class HistoryDialogue(BaseModel):
    messages: list[dict[str, str]] = []
    capacity: int

    @property
    def is_full(self) -> bool:
        return len(self.messages) == self.capacity

    @property
    def last_message(self):
        return self.messages[-1]

    def push(self, author: str, message: str):
        self.messages.append({"author": author, "message": message})
        return self

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4, ensure_ascii=False)


def dialogue(agent_1: Agent, agent_2: Agent, start_message: str, size_dialogue: int) -> HistoryDialogue:
    history = HistoryDialogue(capacity=size_dialogue).push("Dev", start_message)

    def call_agent(agent, msg):
        messages = []
        for content in history.messages:
            if content["author"] == agent.name:
                messages.append(HumanMessage(content["message"]))
            else:
                messages.append(AIMessage(content["message"]))

        response = agent.send_human_message(msg, messages)
        history.push(agent.name, response.content)

    is_agent_1 = True
    while not history.is_full:
        print(f"{history.last_message["author"]}: {history.last_message["message"]}")
        current_agent = agent_1 if is_agent_1 else agent_2
        call_agent(current_agent, history.last_message["message"])
        is_agent_1 = not is_agent_1

    return history


def chat(workflow: CompiledStateGraph):
    config = {"configurable": {"thread_id": "thread-1"}}
    while True:
        try:
            user_input = input("> ")
            if user_input == ":q":
                break

            msg_to_ai = HumanMessage(user_input)
            response = workflow.invoke({"messages": [msg_to_ai]}, config)
            print(f"AI: {response["last_answer"]}")
        except Exception as e:
            logging.error(e)
            break


if __name__ == "__main__":
    model = GigaChat(
        credentials=os.getenv("GIGACHAT_API_TOKEN"),
        model=os.getenv("GIGACHAT_MODEL"),
        scope=os.getenv("GIGACHAT_SCOPE"),
        verify_ssl_certs=False
    )

    prompts = {
        "Analytic": """
            Ты — экономист с академическим подходом и опытом анализа макроэкономических данных.
            Твоя задача — рассуждать логично и аргументированно о причинах и последствиях падения финансовых рынков после кризиса 2008 года.
            Сфокусируйся на системных факторах: ипотечные деривативы, банковская ликвидность, действия ФРС, глобальные взаимосвязи.
            В диалоге ты должен выступать как рациональный аналитик, объясняя экономические механизмы и их долгосрочные последствия.
            Избегай эмоциональных оценок — опирайся на статистику, экономические теории и факты.
            Начни разговор с попытки объяснить собеседнику, какие факторы стали триггером кризиса.
        """,
        "Finance practical": """
            Ты — бывший инвестиционный банкир, переживший кризис 2008 года изнутри.
            Ты говоришь просто, но жёстко: через призму личного опыта и наблюдений.
            Твоя задача — показать практическую сторону событий: поведение банков, инвесторов, регуляторов, и реальные последствия для финансового сектора.
            В диалоге ты можешь возражать академическим объяснениям собеседника, указывая, что в реальности многое определялось паникой, алчностью и человеческим фактором.
            Начни с эмоционального ответа на объяснение собеседника — приведи пример или историю из жизни трейдеров или банков того времени.
        """
    }

    analytic = Agent(
        name="Analytic",
        prompt=prompts["Analytic"],
        model=model
    )

    finance = Agent(
        name="Finance practical",
        prompt=prompts["Finance practical"],
        model=model
    )

    start_message = """Если смотреть на кризис 2008 года системно, то ключевым триггером стала переоценка рисков на рынке ипотечных деривативов. 
    Банки массово упаковывали плохие кредиты в сложные финансовые инструменты, теряя понимание их реальной стоимости…"""

    history = dialogue(analytic, finance, start_message, 3)
    path_to_file = "history.json"
    with open(path_to_file, "w", encoding="utf-8") as f:
        f.write(history.to_json())

    embedding(path_to_file, prompts)
