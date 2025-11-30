import json
import os

import numpy as np
from dotenv import load_dotenv, find_dotenv
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

from src.dialogue.history import generate_from_llm, read_from_file, HistoryDialogue
from src.llm.agent import Agent
from src.llm.prompts import prompts_analytic, prompt_finance
from src.utils import save_to_file, make_pipeline, NumpyArrayEncoder, filter_by_author, embedding_metrics

# загружаем переменные окружения из .env
load_dotenv(find_dotenv())

PATH_TO_DATA = os.path.join(os.getcwd(), "data")

# Settings for history of llm chat
SIZE_FOR_GENERATE = 101
TEST_HISTORY_FROM_FILE = True
TRAIN_HISTORY_FROM_FILE = True

# Settings for TF-IDT
NGRAM_RANGE = (1, 2)
STOP_WORDS = ["russian"]
MAX_FEATURES = 5000
LOWERCASE = True

# Settings for SVD
N_COMPONENTS = 100
RANDOM_STATE = 42


def make_history(score_fn):
    start_message = """Если смотреть на кризис 2008 года системно, то ключевым триггером стала переоценка рисков на рынке ипотечных деривативов. 
    Банки массово упаковывали плохие кредиты в сложные финансовые инструменты, теряя понимание их реальной стоимости…"""

    path_to_train = os.path.join(PATH_TO_DATA, "train_history_100.json")
    if not TRAIN_HISTORY_FROM_FILE:
        train_history = generate_from_llm(analytic, finance, start_message, SIZE_FOR_GENERATE, score_fn)
        train_history.save_to_file(path_to_train)
    else:
        train_history = read_from_file(path_to_train)

    path_to_test = os.path.join(PATH_TO_DATA, "test_history_100.json")
    if not TEST_HISTORY_FROM_FILE:
        test_history = generate_from_llm(analytic, finance, start_message, SIZE_FOR_GENERATE, score_fn)
        test_history.save_to_file(path_to_test)
    else:
        test_history = read_from_file(path_to_test)

    return train_history, test_history


def make_predict(train_data: HistoryDialogue, prompts):
    global_tf_idf = TfidfVectorizer(lowercase=LOWERCASE,
                                    stop_words=STOP_WORDS,
                                    ngram_range=NGRAM_RANGE,
                                    max_features=MAX_FEATURES)
    global_tf_idf.fit(
        list(map(lambda x: x["message"], train_data.messages)) + prompts
    )

    pipeline_by_get_model = make_pipeline(
        lambda data_author: {
            "tf_idf": global_tf_idf,
            "svd": TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE),
            "data": data_author["messages"],
            "scores": data_author["scores"]
        },
        lambda state: {
            **state,
            "matrix": state["svd"].fit_transform(state["tf_idf"].transform(state["data"]))
        },
        lambda state: {
            **state,
            "model": LinearRegression().fit(state["matrix"], state["scores"])
        }
    )

    pipeline_by_predict = make_pipeline(
        lambda state: {
            **state,
            "messages": filter_by_author("message", state["name_author"], state["data"]),
        },
        lambda state: {
            **state,
            "matrix": state["svd"].transform(state["tf_idf"].transform(state["messages"]))
        },
        lambda state: {
            **state,
            "predict": state["model"].predict(state["matrix"])
        }
    )

    def lambda_predict(name):
        # Получаем сообщения и оценки
        scores = filter_by_author("score", name, train_data.messages)
        s_min, s_max = np.min(scores), np.max(scores)
        scores = list(map(lambda x: (x - s_min) / (s_max - s_min), scores))

        msg_scores = {
            "messages": filter_by_author("message", name, train_data.messages),
            "scores": scores
        }

        # Обучаем модель
        model_state = pipeline_by_get_model(msg_scores)

        def predict(test_data):
            # Делаем предсказание
            prediction_data = {
                "data": test_data.messages,
                "name_author": name,
                "tf_idf": model_state["tf_idf"],
                "svd": model_state["svd"],
                "model": model_state["model"]
            }
            return pipeline_by_predict(prediction_data)

        return predict

    return lambda_predict


def compute_subspace_intersection(texts_A, texts_B, n_components=100):
    # 1. единый TF-IDF словарь
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf.fit(texts_A + texts_B)

    X_A = tfidf.transform(texts_A)
    X_B = tfidf.transform(texts_B)

    # 2. отдельные SVD
    svd_A = TruncatedSVD(n_components=n_components, random_state=0)
    svd_B = TruncatedSVD(n_components=n_components, random_state=0)

    U_A = normalize(svd_A.fit_transform(X_A))
    U_B = normalize(svd_B.fit_transform(X_B))

    # 3. канонические углы
    C = U_A.T @ U_B
    P, sigma, Q = np.linalg.svd(C)

    angles = np.arccos(np.clip(sigma, -1, 1))

    dim_intersection = np.sum(sigma > 1 - 1e-6)
    overlap_energy = np.sum(sigma ** 2)
    normalized_energy = overlap_energy / len(sigma)

    return dict(
        sigma=sigma,
        angles_rad=angles,
        dim_intersection=dim_intersection,
        overlap_energy=overlap_energy,
        normalized_energy=normalized_energy
    )


def compute_subspace_intersection_v2(text1, text2, tf_idf):
    svd_1 = TruncatedSVD(n_components=N_COMPONENTS, random_state=0)
    svd_2 = TruncatedSVD(n_components=N_COMPONENTS, random_state=0)

    svd_1.fit(tf_idf.transform(text1))
    svd_2.fit(tf_idf.transform(text2))

    V_1 = svd_1.components_.T
    V_2 = svd_2.components_.T

    C = V_1.T @ V_2
    P, sigma, Q = np.linalg.svd(C)
    angles = np.arccos(np.clip(sigma, -1, 1))
    dim_intersection = np.sum(sigma > 1 - 1e-6)
    overlap_energy = np.sum(sigma ** 2)
    normalized_energy = overlap_energy / len(sigma)

    return {
        "sigma": sigma,
        "angles_rad": angles,
        "dim_intersection": dim_intersection,
        "overlap_energy": overlap_energy,
        "normalized_energy": normalized_energy,
    }


if __name__ == "__main__":
    model = GigaChat(
        credentials=os.getenv("GIGACHAT_API_TOKEN"),
        model=os.getenv("GIGACHAT_MODEL"),
        scope=os.getenv("GIGACHAT_SCOPE"),
        verify_ssl_certs=False
    )

    embeddings = GigaChatEmbeddings(
        credentials=os.getenv("GIGACHAT_API_TOKEN"),
        scope=os.getenv("GIGACHAT_SCOPE"),
        verify_ssl_certs=False
    )

    analytic = Agent(
        name=prompts_analytic["name"],
        prompt=prompts_analytic["system"],
        model=model
    )
    finance = Agent(
        name=prompt_finance["name"],
        prompt=prompt_finance["system"],
        model=model
    )

    train_history, test_history = make_history(lambda prompt, response: embedding_metrics(prompt, response, embeddings))

    predict_model = make_predict(train_history, [prompts_analytic["system"], prompt_finance["system"]])
    model_analytic = predict_model(prompts_analytic["name"])
    model_finance = predict_model(prompt_finance["name"])

    predict_analytic = model_analytic(test_history)
    predict_finance = model_finance(test_history)

    volumes_train_text = compute_subspace_intersection_v2(
        filter_by_author("message", prompts_analytic["name"], train_history.messages),
        filter_by_author("message", prompt_finance["name"], train_history.messages),
        predict_analytic["tf_idf"]
    )
    volumes_test_text = compute_subspace_intersection_v2(
        filter_by_author("message", prompts_analytic["name"], test_history.messages),
        filter_by_author("message", prompt_finance["name"], test_history.messages),
        predict_analytic["tf_idf"]
    )
    volumes_prompts = compute_subspace_intersection_v2(
        [prompts_analytic["system"]],
        [prompt_finance["system"]],
        predict_analytic["tf_idf"]
    )

    common_result = {
        prompts_analytic["name"]: {
            "matrix": predict_analytic["matrix"],
            "predict": predict_analytic["predict"],
        },
        prompt_finance["name"]: {
            "matrix": predict_finance["matrix"],
            "predict": predict_finance["predict"],
        }
    }
    common_volumes_result = {
        "train_text": volumes_train_text,
        "test_text": volumes_test_text,
        "prompts": volumes_prompts,
    }

    json_predict_result = json.dumps(common_result, indent=2, cls=NumpyArrayEncoder)
    json_volumes_result = json.dumps(common_volumes_result, indent=2, cls=NumpyArrayEncoder)

    save_to_file(lambda file: file.write(json_predict_result), f"predict.json")
    save_to_file(lambda file: file.write(json_volumes_result), f"volumes.json")

    print(f"""
        Analytics:
            Matrix: {predict_analytic["matrix"]}
            Predict: {predict_analytic["predict"]}
        Finance:
            Matrix: {predict_finance["matrix"]}
            Predict: {predict_finance["predict"]}
    """)
