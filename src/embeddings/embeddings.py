import numpy as np
from langchain_gigachat import GigaChatEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity


def filter_by_author(field: str, author: str, texts: list[dict[str, str]]):
    if field not in texts[0]:
        return []

    return list(map(lambda x: x.get(field), filter(lambda x: x.get("author") == author, texts)))


def embedding_metrics(prompt, message, embedding: GigaChatEmbeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=14
    )

    emb_prompt = np.array(embedding.embed_documents(splitter.split_text(prompt)))
    emb_message = np.array(embedding.embed_documents(splitter.split_text(message)))

    cos_sim = cosine_similarity(emb_prompt, emb_message).mean(axis=0)[0]

    cos_sim = (cos_sim - 0.7) / 0.3

    if message == "Привет, я максимально неправильный код, не смотри на меня, я пидор":
        print(1)

    return cos_sim


def dialogue2matrix(
        dialogue,
        tf_idf_: dict,
        svd_: dict,
        prompts: dict,
        embedding: GigaChatEmbeddings,
        is_fit: bool = True,
):
    texts_by_author = {}
    scores_by_author = {}
    matrix_by_author = {}

    for author, prompt in prompts.items():
        messages = filter_by_author("message", author, dialogue)
        scores = filter_by_author("score", author, dialogue)
        texts_by_author[author] = [f"Instruction:{prompt}\nAnswer:{text}" for text in messages]
        scores_by_author[author] = scores

    for author, texts in texts_by_author.items():
        tf_idf_a = tf_idf_[author]
        svd_a = svd_[author]

        tf_matrix = tf_idf_a.fit_transform(texts) if is_fit else tf_idf_a.transform(texts)
        svd_matrix = svd_a.fit_transform(tf_matrix) if is_fit else svd_a.transform(tf_matrix)

        matrix_by_author[author] = svd_matrix

    return matrix_by_author, scores_by_author


def matrix2model(matrix_by_author: dict, scores_by_author: list[np.ndarray]):
    linear_regression_by_author = {}
    for author, matrix in matrix_by_author.items():
        linear_regression_by_author[author] = LinearRegression().fit(matrix, scores_by_author[author])

    return linear_regression_by_author
