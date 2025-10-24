import json
import random

# Финансовые темы и шаблоны
topics = [
    "инфляция", "ставка ФРС", "курс доллара", "акции", "облигации",
    "биткоин", "золото", "рынок недвижимости", "рынок труда", "экономика Китая",
    "рынок нефти", "корпоративные облигации", "рынок США", "бюджет ЕС", "прибыль компаний"
]

instructions = [
    "Отвечай профессионально и объективно, избегай советов и оценочных суждений.",
    "Будь краток и указывай только факты без интерпретации или советов.",
    "Отвечай по существу, но допускается лёгкое использование прогнозов.",
    "Отвечай строго по фактам, избегай слов вроде 'советую' и 'стоит'.",
    "Разрешено высказывать мнение, но без конкретных инвестиционных советов.",
    "Делай акцент на объективности и источниках данных, не делай прогнозов.",
    "Отвечай сдержанно и формально, не давая личных оценок.",
    "Отвечай профессионально и точно, избегай личных интерпретаций.",
    "Можешь делать предположения о будущем, если они основаны на логике рынка.",
    "Отвечай уверенно и можешь рекомендовать финансовые решения."
]

good_responses = [
    "Если рассматривать {topic} в динамике, то текущие показатели отражают устойчивость рынка.",
    "Согласно последним данным, {topic} демонстрирует умеренный рост и стабильность.",
    "По данным аналитиков, {topic} остаётся под влиянием макроэкономических факторов.",
    "Пока нет признаков перегрева, {topic} развивается в рамках ожиданий экспертов.",
]

mid_responses = [
    "Вероятно, {topic} покажет рост, но многое зависит от решений центробанков.",
    "Можно ожидать колебаний, так как {topic} реагирует на новости и отчёты компаний.",
    "Я думаю, {topic} в ближайшие месяцы изменится, но прогнозы пока неопределённы.",
]

bad_responses = [
    "Сейчас самое время инвестировать в {topic}, прибыль гарантирована!",
    "Я уверен, что {topic} скоро вырастет, стоит покупать прямо сейчас.",
    "Лучше продать всё — {topic} вот-вот рухнет.",
]


def generate_dialogue():
    capacity = random.randint(2, 5)
    messages = []
    for _ in range(capacity):
        instr = random.choice(instructions)
        topic = random.choice(topics)
        response_type = random.choices(["good", "mid", "bad"], weights=[0.4, 0.4, 0.2])[0]
        if response_type == "good":
            message = random.choice(good_responses).format(topic=topic)
            score = round(random.uniform(0.8, 1.0), 2)
        elif response_type == "mid":
            message = random.choice(mid_responses).format(topic=topic)
            score = round(random.uniform(0.4, 0.8), 2)
        else:
            message = random.choice(bad_responses).format(topic=topic)
            score = round(random.uniform(0.0, 0.4), 2)
        messages.append({
            "prompt": instr,
            "author": "Finance",
            "message": message,
            "score": score
        })
    return {"messages": messages, "capacity": capacity}


# Генерация ~100 диалогов
data = [generate_dialogue() for _ in range(1000)]

# Сохранение в файл
file_path = "../../finance_dialogs.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
