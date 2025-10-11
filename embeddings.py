import json

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity(matrix):
    sim_msg = []
    for i in range(1, matrix.shape[0]):
        sim = cosine_similarity(matrix[0], matrix[i])
        sim_msg.append(sim[0])
    return sim_msg


def get_messages_by_author(author: str, texts: list):
    return list(map(lambda x: x.get("message"), filter(lambda x: x.get("author") == author, texts)))


def processed(author, prompt, texts) -> dict:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=[],  # можно 'english' если тексты на английском
        ngram_range=(1, 4),  # можно добавить биграммы для контекста
        max_features=5000
    )
    responses = get_messages_by_author(author, texts)
    documents = [prompt] + responses
    matrices = vectorizer.fit_transform(documents)
    similarity = get_similarity(matrices)

    return {
        "vectorizer": vectorizer,
        "responses": responses,
        "documents": documents,
        "matrix": matrices,
        "similarity": similarity,
    }


def plot_cloud(vectorizer, matrix):
    doc_idx = 0
    feature_names = vectorizer.get_feature_names_out()
    tfidf_values = matrix[doc_idx].toarray()[0]

    # Преобразуем в 2D через PCA для визуализации
    pca = PCA(n_components=2)
    tfidf_2d = pca.fit_transform(matrix.T)  # транспонируем, чтобы слова были объектами

    # Облако точек
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(feature_names):
        x, y = tfidf_2d[i]
        plt.scatter(x, y, s=tfidf_values[i] * 1000 + 10, color='skyblue')  # размер пропорционален TF-IDF
        plt.text(x, y, word, fontsize=10)

    plt.title(f"Облако слов (TF-IDF) для документа")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.show()


def save_to_json(path_to_file, data: dict):
    json_ = {}
    for key, value in data.items():
        json_[key] = {
            "prompt": value["documents"][0],
            "responses": [{"response": response, "similarity": similarity[0]} for response, similarity in
                          zip(value["responses"], value["similarity"])],
            "matrix": value["matrix"].toarray().tolist(),
        }

    with open(path_to_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(json_, ensure_ascii=False))


def embedding(path_to_file, prompts):
    file = open(path_to_file, "r", encoding="utf-8")
    texts = json.load(file).get("messages")
    file.close()

    results = {author: processed(author, prompt, texts) for author, prompt in prompts.items()}
    save_to_json("result.json", results)
