import os

from dotenv import load_dotenv, find_dotenv
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.dialogue.history import generate_from_llm, read_from_file
from src.embeddings.embeddings import dialogue2matrix, matrix2model, dialogue2matrix
from src.llm.agent import Agent
from src.llm.prompts import prompts_analytic, prompt_finance

# загружаем переменные окружения из .env
load_dotenv(find_dotenv())

HISTORY_FROM_FILE = True

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

    prompts = {
        prompts_analytic["name"]: prompts_analytic["system"],
        prompt_finance["name"]: prompt_finance["system"],
    }

    tf_idt = {
        prompts_analytic["name"]: TfidfVectorizer(
            lowercase=True,
            stop_words=['russian'],
            ngram_range=(1, 2),  # можно добавить биграммы для контекста
            max_features=5000
        ),
        prompt_finance["name"]: TfidfVectorizer(
            lowercase=True,
            stop_words=['russian'],
            ngram_range=(1, 2),  # можно добавить биграммы для контекста
            max_features=5000
        )
    }

    svd = {
        prompts_analytic["name"]: TruncatedSVD(n_components=100, random_state=42),
        prompt_finance["name"]: TruncatedSVD(n_components=100, random_state=42)
    }

    start_message = """Если смотреть на кризис 2008 года системно, то ключевым триггером стала переоценка рисков на рынке ипотечных деривативов. 
    Банки массово упаковывали плохие кредиты в сложные финансовые инструменты, теряя понимание их реальной стоимости…"""

    path_to_train = "train_history.json"
    if not HISTORY_FROM_FILE:
        train_history = generate_from_llm(analytic, finance, start_message, 10)
        train_history.save_to_file(path_to_train)
    else:
        train_history = read_from_file(path_to_train)

    path_to_test = "test_history.json"
    if not HISTORY_FROM_FILE:
        test_history = generate_from_llm(analytic, finance, start_message, 10)
        test_history.save_to_file(path_to_test)
    else:
        test_history = read_from_file(path_to_test)

    matrices_train, scores_train = dialogue2matrix(
        train_history.messages,
        tf_idt,
        svd,
        {name: prompt for name, prompt in prompts.items()},
        embedding
    )
    models = matrix2model(matrices_train, scores_train)

    matrices_test, scores_test = dialogue2matrix(
        test_history.messages,
        tf_idt,
        svd,
        {name: prompt for name, prompt in prompts.items()},
        embedding,
        False
    )
    predict_by_author = {}
    for author, model in models.items():
        predict_by_author[author] = model.predict(matrices_test[author])

    print(predict_by_author)
