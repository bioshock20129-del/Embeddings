from collections.abc import Callable
from json import JSONEncoder

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_gigachat import GigaChatEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity


def save_to_file(fn: Callable, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        fn(f)


def make_pipeline(*steps):
    def wrapper(*args, **kwargs):
        result = steps[0](*args, **kwargs)
        for step in steps[1:]:
            result = step(result)
        return result

    return wrapper


def embedding_metrics(prompt, message, embedding: Embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=14
    )

    emb_prompt = np.array(embedding.embed_documents(splitter.split_text(prompt)))
    emb_message = np.array(embedding.embed_documents(splitter.split_text(message)))

    cos_sim = cosine_similarity(emb_prompt, emb_message).mean(axis=0)[0]

    return cos_sim


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def filter_by_author(field: str, author: str, texts: list[dict[str, str]]):
    if not texts or field not in texts[0]:
        return []
    return list(map(lambda x: x.get(field), filter(lambda x: x.get("author") == author, texts)))
