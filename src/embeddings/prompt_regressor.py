import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression


def principal_angle(U1: np.ndarray, U2: np.ndarray) -> tuple[float, float]:
    """
    U1, U2 – (d, 1), нормированные базисы двух подпространств.
    Возвращает (sigma, angle).
    """
    C = U1.T @ U2
    sigma = float(np.clip(np.linalg.svd(C, compute_uv=False)[0], 0, 1))
    angle = float(np.arccos(sigma))
    return sigma, angle


class PromptAngleRegressor(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Класс, похожий на TruncatedSVD+LinearRegression, но:
    - учитывает подпространство промпта,
    - добавляет признаки угла prompt↔answer,
    - обучает линейную регрессию на расширенных признаках.
    """

    def __init__(
            self,
            prompt: str,
            tf_idf: TfidfVectorizer,
            svd: TruncatedSVD,
    ):
        self.prompt = prompt

        # Компоненты модели (после fit)
        self.tfidf_ = tf_idf
        self.svd_ = svd

        self.U_prompt_ = None
        self.model_ = None

    def _answer_to_features(self, answer: str) -> np.ndarray:
        """TF-IDF → SVD → (sigma, angle) → финальный вектор признаков."""
        a_tfidf = self.tfidf_.transform([answer])
        a_emb = self.svd_.transform(a_tfidf)[0]  # (d,)

        # 1D подпространство ответа
        U_answer = a_emb.reshape(-1, 1)
        U_answer = normalize(U_answer, axis=0)

        sigma, angle = principal_angle(self.U_prompt_, U_answer)
        return np.concatenate([a_emb, [sigma, angle]])

    # ---------------------------------------------
    # fit
    # ---------------------------------------------
    def fit(self, answers: list[str], scores: list[float]):
        """
        Строит TF-IDF, SVD, подпространство промпта и обучает регрессию.
        """
        # 1. TF-IDF
        self.tfidf_.fit([self.prompt] + answers)

        P_tfidf = self.tfidf_.transform([self.prompt])

        # 2. SVD
        self.svd_.fit(self.tfidf_.transform([self.prompt] + answers))

        P_emb = self.svd_.transform(P_tfidf)[0]

        # 1D подпространство промпта
        self.U_prompt_ = P_emb.reshape(-1, 1)
        self.U_prompt_ = normalize(self.U_prompt_, axis=0)

        # 3. Формирование X
        X = np.vstack([self._answer_to_features(a) for a in answers])
        y = np.array(scores, dtype=float)

        # 4. Регрессия
        self.model_ = LinearRegression()
        self.model_.fit(X, y)

        return self

    # ---------------------------------------------
    # transform (как у SVD)
    # ---------------------------------------------
    def transform(self, answers: list[str]) -> np.ndarray:
        """
        Возвращает матрицу признаков для новых ответов:
        [SVD embedding..., sigma, angle]
        """
        return np.vstack([self._answer_to_features(a) for a in answers])

    # ---------------------------------------------
    # predict (как у регрессора)
    # ---------------------------------------------
    def predict(self, answers: list[str]) -> np.ndarray:
        """
        Предсказывает качество ответа по признакам.
        """
        X = self.transform(answers)
        return self.model_.predict(X)
