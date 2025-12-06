from collections.abc import Callable

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRegressor:
    name: str = "EmbeddingRegressor"

    def __init__(
            self,
            embedding: Callable[[list[str]], list],
            chunk_size=500,
            chunk_overlap=14
    ):
        self.embedding = embedding
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.emb_prompt = np.array([])

    def _get_embedding(self, content):
        split = self.splitter.split_text(content)
        embedding = self.embedding(split)
        return np.array(embedding)

    def fit(self, prompt):
        self.emb_prompt = self._get_embedding(prompt)
        return self

    def predict(self, answers: list) -> np.ndarray:
        result = []
        for i in range(len(answers)):
            print(f"{self.name}: Proccessing answer {i} of {len(answers)}")
            answer = answers[i]
            emb_answer = self._get_embedding(answer)
            cos_sim = cosine_similarity(self.emb_prompt, emb_answer).mean(axis=0)[0]
            result.append(cos_sim)

        return np.array(result)

    @staticmethod
    def get_and_fit(embedding: Callable[[list[str]], list], prompt: str):
        model = EmbeddingRegressor(embedding=embedding)
        model.fit(prompt)
        return model
