import logging
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import AnyMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph

from agent import Agent


class State(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    last_answer: str


def make_graph(agent: Agent):
    def invoke_agent(state: State):
        try:
            messages = state.get("messages", [])
            if len(messages) == 0:
                raise ValueError("No messages")

            response = agent.send_human_message(messages[-1].content)
            return {**state, "messages": [response], "last_answer": response.content}
        except Exception as e:
            logging.error(e)
            msg = AIMessage("Произошла ошибка, повторите попытку позже")
            return {**state, "messages": [msg], "last_answer": msg.content}

    return (
        StateGraph(State)
        .add_node("agent", invoke_agent)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile(checkpointer=InMemorySaver())
    )
