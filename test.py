import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Пример текста
documents = [
    "кошка сидит на окне",
    "собака лежит на полу",
    "кошка и собака друзья"
]

# Создаём TF-IDF векторизатор
vectorizer = TfidfVectorizer(lowercase=True, stop_words=None, ngram_range=(1, 1))
tfidf_matrix = vectorizer.fit_transform(documents)

# Выбираем документ для визуализации
doc_idx = 0
feature_names = vectorizer.get_feature_names_out()
tfidf_values = tfidf_matrix[doc_idx].toarray()[0]

# Преобразуем в 2D через PCA для визуализации
pca = PCA(n_components=2)
tfidf_2d = pca.fit_transform(tfidf_matrix.T)  # транспонируем, чтобы слова были объектами

# Облако точек
plt.figure(figsize=(8, 6))
for i, word in enumerate(feature_names):
    x, y = tfidf_2d[i]
    plt.scatter(x, y, s=tfidf_values[i] * 1000 + 10, color='skyblue')  # размер пропорционален TF-IDF
    plt.text(x + 0.01, y + 0.01, word, fontsize=10)

plt.title(f"Облако слов (TF-IDF) для документа: '{documents[doc_idx]}'")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.show()
