import re

import numpy as np

from src.llm.agent import Agent


class LLMRegressor:
    name: str = "LLMRegressor"

    def __init__(self, model: Agent):
        self.model = model

    def fit(self, prompt):
        self.model.bind({
            "system_prompt": prompt,
        })

        return self

    def predict(self, answers: list) -> np.ndarray:
        result = []
        for i in range(len(answers)):
            print(f"{self.name}: Proccessing answer {i} of {len(answers)}")
            answer = answers[i]
            res = self.model.send_human_message(answer).content
            match = re.findall(r"(\d+)", res)
            score = float(match[0]) / 100 if match else 0

            result.append(score)



        return np.array(result)

    @staticmethod
    def get_and_fit(model: Agent, prompt: str):
        model = LLMRegressor(model=model)
        model.fit(prompt)
        return model
