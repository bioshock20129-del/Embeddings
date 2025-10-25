import json
import os

from dotenv import load_dotenv, find_dotenv
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

from src.dialogue.history import generate_from_llm, read_from_file
from src.llm.agent import Agent
from src.llm.prompts import prompts_analytic, prompt_finance
from src.utils import save_to_file, make_pipeline, NumpyArrayEncoder, filter_by_author

# загружаем переменные окружения из .env
load_dotenv(find_dotenv())

TEST_HISTORY_FROM_FILE = False
TRAIN_HISTORY_FROM_FILE = True


def make_history():
    start_message = """Если смотреть на кризис 2008 года системно, то ключевым триггером стала переоценка рисков на рынке ипотечных деривативов. 
    Банки массово упаковывали плохие кредиты в сложные финансовые инструменты, теряя понимание их реальной стоимости…"""

    path_to_train = "train_history.json"
    if not TRAIN_HISTORY_FROM_FILE:
        train_history = generate_from_llm(analytic, finance, start_message, 10)
        train_history.save_to_file(path_to_train)
    else:
        train_history = read_from_file(path_to_train)

    path_to_test = "test_history_v2.json"
    if not TEST_HISTORY_FROM_FILE:
        test_history = generate_from_llm(analytic, finance, start_message, 100)
        test_history.save_to_file(path_to_test)
    else:
        test_history = read_from_file(path_to_test)

    return train_history, test_history


def make_predict():
    pipeline_by_get_model = make_pipeline(
        lambda data_author: {
            "tf_idf": TfidfVectorizer(lowercase=True, stop_words=['russian'], ngram_range=(1, 2), max_features=5000),
            "svd": TruncatedSVD(n_components=100, random_state=42),
            "data": data_author["messages"],
            "scores": data_author["scores"]
        },
        lambda state: {
            **state,
            "matrix": state["svd"].fit_transform(state["tf_idf"].fit_transform(state["data"]))
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

    def lambda_predict(train_data, name):
        # Получаем сообщения и оценки
        msg_scores = {
            "messages": filter_by_author("message", name, train_data.messages),
            "scores": filter_by_author("score", name, train_data.messages)
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


def test_predict():
    model_analytic = make_predict()(train_history, prompts_analytic["name"])
    model_finance = make_predict()(train_history, prompt_finance["name"])

    predict_analytic = model_analytic(test_history)
    predict_finance = model_finance(test_history)

    print(f"""
        Analytics:
            Matrix: {predict_analytic["matrix"]}
            Predict: {predict_analytic["predict"]}
        Finance:
            Matrix: {predict_finance["matrix"]}
            Predict: {predict_finance["predict"]}
    """)

    print(1)


if __name__ == "__main__":
    model = GigaChat(
        credentials=os.getenv("GIGACHAT_API_TOKEN"),
        model=os.getenv("GIGACHAT_MODEL"),
        scope=os.getenv("GIGACHAT_SCOPE"),
        verify_ssl_certs=False
    )
    embedding = GigaChatEmbeddings(
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

    train_history, test_history = make_history()
    model_analytic = make_predict()(train_history, prompts_analytic["name"])
    model_finance = make_predict()(train_history, prompt_finance["name"])

    predict_analytic = model_analytic(test_history)
    predict_finance = model_finance(test_history)

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

    json_ = json.dumps(common_result, indent=2, cls=NumpyArrayEncoder)
    save_to_file(lambda file: file.write(json_), f"result.json")

    print(f"""
        Analytics:
            Matrix: {predict_analytic["matrix"]}
            Predict: {predict_analytic["predict"]}
        Finance:
            Matrix: {predict_finance["matrix"]}
            Predict: {predict_finance["predict"]}
    """)
