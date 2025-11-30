def generate_dot_workflow():
    """Генерирует DOT код для общей схемы работы программы"""

    dot_code = """
digraph ProgramWorkflow {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor="lightblue", fontname="Arial"];
    edge [fontname="Arial", fontsize=10];

    // Блок 1: Подготовка данных
    subgraph cluster_1 {
        label = "БЛОК 1: ПОДГОТОВКА ДАННЫХ";
        style=filled;
        fillcolor=lightgrey;

        env [label="Загрузка переменных окружения", fillcolor="lightblue"];
        init [label="Инициализация GigaChat и агентов", fillcolor="lightblue"];
        decision [label="Использовать существующую историю?", fillcolor="gold"];

        env -> init;
        init -> decision;
    }

    // Ветвление для тренировочных данных
    subgraph cluster_train {
        label = "Тренировочные данные";
        style=dashed;

        train_gen [label="Генерация тренировочной истории", fillcolor="lightgreen"];
        train_load [label="Загрузка тренировочной истории", fillcolor="lightgreen"];
    }

    // Ветвление для тестовых данных  
    subgraph cluster_test {
        label = "Тестовые данные";
        style=dashed;

        test_gen [label="Генерация тестовой истории", fillcolor="lightgreen"];
        test_load [label="Загрузка тестовой истории", fillcolor="lightgreen"];
    }

    // Блок 2: ML-пайплайн
    subgraph cluster_2 {
        label = "БЛОК 2: ML-ПАЙПЛАЙН";
        style=filled;
        fillcolor=lightgrey;

        pipeline [label="Построение пайплайна: TF-IDF → SVD → LinearRegression", fillcolor="orange"];
        train_model [label="Обучение модели на тренировочных данных", fillcolor="orange"];

        pipeline -> train_model;
    }

    // Блок 3: Анализ и сохранение
    subgraph cluster_3 {
        label = "БЛОК 3: АНАЛИЗ И СОХРАНЕНИЕ";
        style=filled;
        fillcolor=lightgrey;

        predict [label="Предсказание оценок на тестовых данных", fillcolor="pink"];
        analysis [label="Анализ пересечения подпространств", fillcolor="pink"];
        save [label="Сохранение результатов в JSON", fillcolor="pink"];
        end [label="Завершение работы", fillcolor="pink"];

        predict -> analysis;
        analysis -> save;
        save -> end;
    }

    // Связи между блоками
    decision -> train_gen [label="Нет"];
    decision -> train_load [label="Да"];
    decision -> test_gen [label="Нет"];
    decision -> test_load [label="Да"];

    train_gen -> test_gen;
    train_load -> test_load;

    test_gen -> pipeline;
    test_load -> pipeline;

    train_model -> predict;
}
"""
    return dot_code


def generate_dot_pipeline():
    """Генерирует DOT код для детальной схемы ML-пайплайна"""

    dot_code = """
digraph MLPipeline {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="lightblue", fontname="Arial"];
    edge [fontname="Arial", fontsize=10];

    // Входные данные
    input_data [label="Входные данные:\\n• Тренировочные диалоги\\n• Тестовые диалоги\\n• Системные промпты", fillcolor="lightgrey"];

    // Основные этапы обработки
    tfidf [label="TF-IDF\\nВекторизация текстов", fillcolor="lightgreen"];
    svd [label="SVD\\nСнижение размерности", fillcolor="lightgreen"]; 
    lr_train [label="Linear Regression\\nОбучение модели", fillcolor="orange"];
    predict [label="Предсказание\\nоценок качества", fillcolor="orange"];
    subspace [label="Анализ пересечения\\nподпространств", fillcolor="pink"];

    // Выходные данные
    output [label="Выходные данные:\\n• predict.json\\n• volumes.json\\n• Визуализации", fillcolor="lightgrey"];

    // Связи
    input_data -> tfidf;
    tfidf -> svd;
    svd -> lr_train;
    lr_train -> predict;
    predict -> subspace;
    subspace -> output;
}
"""
    return dot_code


def generate_dot_subspace_analysis():
    """Генерирует DOT код для схемы анализа пересечения подпространств"""

    dot_code = """
digraph SubspaceAnalysis {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor="lightblue", fontname="Arial"];
    edge [fontname="Arial", fontsize=10];

    // Входные данные
    texts_a [label="Тексты агента A", fillcolor="lightgreen"];
    texts_b [label="Тексты агента B", fillcolor="lightgreen"];

    // Процесс анализа
    tfidf [label="Общий TF-IDF\\nвекторизатор", fillcolor="orange"];
    svd_a [label="SVD для\\nагента A", fillcolor="orange"];
    svd_b [label="SVD для\\nагента B", fillcolor="orange"];
    components_a [label="Матрица компонент V_A", fillcolor="pink"];
    components_b [label="Матрица компонент V_B", fillcolor="pink"];
    covariance [label="Матрица ковариации\\nC = V_A^T V_B", fillcolor="purple", fontcolor="white"];
    svd_c [label="SVD матрицы C\\nC = P Σ Q^T", fillcolor="purple", fontcolor="white"];

    // Метрики
    metrics [label="Метрики анализа:\\n• Сингулярные числа σ_i\\n• Канонические углы θ_i\\n• Размер пересечения\\n• Энергия перекрытия", fillcolor="lightgrey"];

    // Связи
    texts_a -> tfidf;
    texts_b -> tfidf;
    tfidf -> svd_a;
    tfidf -> svd_b;
    svd_a -> components_a;
    svd_b -> components_b;
    components_a -> covariance;
    components_b -> covariance;
    covariance -> svd_c;
    svd_c -> metrics;
}
"""
    return dot_code


# Сохраняем DOT файлы
if __name__ == "__main__":
    print("Генерация DOT файлов...")

    # Общая схема работы программы
    with open('program_workflow.dot', 'w', encoding='utf-8') as f:
        f.write(generate_dot_workflow())
    print("Создан файл: program_workflow.dot")

    # Схема ML-пайплайна
    with open('ml_pipeline.dot', 'w', encoding='utf-8') as f:
        f.write(generate_dot_pipeline())
    print("Создан файл: ml_pipeline.dot")

    # Схема анализа пересечения подпространств
    with open('subspace_analysis.dot', 'w', encoding='utf-8') as f:
        f.write(generate_dot_subspace_analysis())
    print("Создан файл: subspace_analysis.dot")

    print("\nИнструкция:")
    print("1. Перейдите на сайт: https://edotor.net/")
    print("2. Скопируйте содержимое .dot файла в левое окно")
    print("3. Настройте внешний вид при необходимости")
    print("4. Скачайте изображение в нужном формате (PNG/SVG)")
    print("5. Вставьте изображение в диплом")