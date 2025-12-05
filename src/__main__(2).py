import json
import os

import numpy as np
from dotenv import load_dotenv, find_dotenv
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.dialogue.history import generate_from_llm, read_from_file
from src.embeddings.prompt_regressor_v2 import PromptAngleRegressorAdvanced
from src.llm.agent import Agent
from src.llm.prompts import prompts_analytic, prompt_finance
from src.utils import filter_by_author, embedding_metrics, save_to_file, NumpyArrayEncoder

# загружаем переменные окружения из .env
load_dotenv(find_dotenv())

PATH_TO_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Settings for history of llm chat
SIZE_FOR_GENERATE = 101
TEST_HISTORY_FROM_FILE = True
TRAIN_HISTORY_FROM_FILE = True

TRAIN_FILE = "train_history_100.json"
TEST_FILE = "test_history_100.json"

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

    path_to_train = os.path.join(PATH_TO_DATA, TRAIN_FILE)
    if not TRAIN_HISTORY_FROM_FILE:
        train_history = generate_from_llm(analytic, finance, start_message, SIZE_FOR_GENERATE, score_fn)
        train_history.save_to_file(path_to_train)
    else:
        train_history = read_from_file(path_to_train)

    path_to_test = os.path.join(PATH_TO_DATA, TEST_FILE)
    if not TEST_HISTORY_FROM_FILE:
        test_history = generate_from_llm(analytic, finance, start_message, SIZE_FOR_GENERATE, score_fn)
        test_history.save_to_file(path_to_test)
    else:
        test_history = read_from_file(path_to_test)

    return train_history, test_history


def predictor(
        name: str,
        prompt: str,
        train_data: list[dict]
):
    tf_idf_ = TfidfVectorizer(lowercase=LOWERCASE, stop_words=STOP_WORDS, ngram_range=NGRAM_RANGE,
                              max_features=MAX_FEATURES)
    svd_ = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    predictor_ = PromptAngleRegressorAdvanced(
        prompt=prompt,
        answer_subspace_dim=5,
    )

    messages_analytic = filter_by_author("message", name, train_data)
    scores_analytic = filter_by_author("score", name, train_data)
    s_min, s_max = np.min(scores_analytic), np.max(scores_analytic)
    scores_analytic = list(map(lambda x: (x - s_min) / (s_max - s_min), scores_analytic))

    predictor_.fit(answers=messages_analytic, scores=scores_analytic)
    return predictor_


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

    predictor_analytic = predictor(prompts_analytic["name"], prompts_analytic["system"], train_history.messages)
    predictor_finance = predictor(prompt_finance["name"], prompt_finance["system"], train_history.messages)

    analytic_answers = filter_by_author("message", prompts_analytic["name"], test_history.messages)
    finance_answers = filter_by_author("message", prompt_finance["name"], test_history.messages)

    print(f"Analytic: {predictor_analytic.predict(analytic_answers)}")
    print(f"Finance: {predictor_finance.predict(finance_answers)}")

    predict_result = {
        prompts_analytic["name"]: {
            "Predict scores": predictor_analytic.predict(analytic_answers),
            "Explains": [{"answer": answer, **predictor_analytic.explain(answer)} for answer in analytic_answers]
        },
        prompt_finance["name"]: {
            "Predict scores": predictor_finance.predict(finance_answers),
            "Explains": [{"answer": answer, **predictor_finance.explain(answer)} for answer in finance_answers]
        }
    }

    json_result = json.dumps(predict_result, indent=2, cls=NumpyArrayEncoder, ensure_ascii=False)
    save_to_file(lambda file: file.write(json_result), f"../results.json")


