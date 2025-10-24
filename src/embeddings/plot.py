from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


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
