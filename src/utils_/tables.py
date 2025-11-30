import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


def create_summary_table(data):
    """Создает сводную таблицу результатов"""

    train_data = data['train_text']
    test_data = data['test_text']
    prompts_data = data['prompts']

    # Подготовка данных для таблицы
    metrics = [
        ("Нормализованная энергия",
         f"{train_data['normalized_energy']:.3f}",
         f"{test_data['normalized_energy']:.3f}",
         f"{prompts_data['normalized_energy']:.3f}"),

        ("Размер пересечения",
         f"{train_data['dim_intersection']}",
         f"{test_data['dim_intersection']}",
         f"{prompts_data['dim_intersection']}"),

        ("Энергия перекрытия",
         f"{train_data['overlap_energy']:.3f}",
         f"{test_data['overlap_energy']:.3f}",
         f"{prompts_data['overlap_energy']:.3f}"),

        ("Максимальное σ",
         f"{max(train_data['sigma']):.3f}",
         f"{max(test_data['sigma']):.3f}",
         f"{max(prompts_data['sigma']):.3f}"),

        ("Минимальный угол (°)",
         f"{np.degrees(min(train_data['angles_rad'])):.1f}",
         f"{np.degrees(min(test_data['angles_rad'])):.1f}",
         f"{np.degrees(min(prompts_data['angles_rad'])):.1f}")
    ]

    # Создание таблицы
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    table_data = [['Метрика', 'Обучающие данные', 'Тестовые данные', 'Системные промпты']]
    for metric in metrics:
        table_data.append(list(metric))

    # Создание таблицы с разными цветами для заголовка и строк
    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.24])

    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Цвета для строк
    header_color = '#4A708B'
    row_colors = ['#E8E8E8', '#F5F5F5']

    # Применение стилей
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:  # Заголовок
                table[(i, j)].set_facecolor(header_color)
                table[(i, j)].set_text_props(color='white', weight='bold')
            else:
                table[(i, j)].set_facecolor(row_colors[i % 2])

    plt.title('Сводная таблица анализа пересечения подпространств стилей агентов',
              fontsize=14, fontweight='bold', pad=20)

    return fig, table


def create_detailed_metrics_table(data):
    """Создает таблицу с дополнительными метриками"""

    train_data = data['train_text']
    test_data = data['test_text']

    # Вычисление дополнительных метрик
    train_high_corr = sum(1 for s in train_data['sigma'] if s > 0.5)
    test_high_corr = sum(1 for s in test_data['sigma'] if s > 0.5)

    train_avg_angle = np.mean(train_data['angles_rad'])
    test_avg_angle = np.mean(test_data['angles_rad'])

    metrics = [
        ("Компоненты с σ > 0.5", f"{train_high_corr}", f"{test_high_corr}", "-"),
        ("Средний угол (°)", f"{np.degrees(train_avg_angle):.1f}", f"{np.degrees(test_avg_angle):.1f}", "-"),
        ("Количество компонент", f"{len(train_data['sigma'])}", f"{len(test_data['sigma'])}", "-"),
        ("Эффективная размерность*", f"{len([s for s in train_data['sigma'] if s > 0.1])}",
         f"{len([s for s in test_data['sigma'] if s > 0.1])}", "-")
    ]

    # Создание таблицы
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    table_data = [['Дополнительная метрика', 'Обучающие данные', 'Тестовые данные', 'Примечание']]
    for metric in metrics:
        table_data.append(list(metric))

    table = ax.table(cellText=table_data,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.25])

    # Стилизация
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    header_color = '#2E8B57'
    row_colors = ['#F0F8FF', '#E6E6FA']

    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:
                table[(i, j)].set_facecolor(header_color)
                table[(i, j)].set_text_props(color='white', weight='bold')
            else:
                table[(i, j)].set_facecolor(row_colors[i % 2])

    plt.title('Дополнительные метрики анализа подпространств',
              fontsize=12, fontweight='bold', pad=20)

    # Добавляем примечание
    plt.figtext(0.1, 0.02, '*Эффективная размерность: количество компонент с σ > 0.1',
                fontsize=9, style='italic')

    return fig, table


# Основной код
if __name__ == "__main__":
    # Загрузка данных
    try:
        with open('../volumes.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Файл volumes.json не найден!")
        exit()

    # Создание основной сводной таблицы
    print("Создание сводной таблицы...")
    fig1, table1 = create_summary_table(data)
    plt.tight_layout()
    plt.savefig('summary_table.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Создание таблицы с дополнительными метриками
    print("Создание таблицы с дополнительными метриками...")
    fig2, table2 = create_detailed_metrics_table(data)
    plt.tight_layout()
    plt.savefig('detailed_metrics_table.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # Вывод в консоль для проверки
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"{'Метрика':<25} {'Обучающие':<12} {'Тестовые':<12} {'Промпты':<12}")
    print("-" * 80)

    train_data = data['train_text']
    test_data = data['test_text']
    prompts_data = data['prompts']

    metrics_console = [
        ("Нормализованная энергия",
         f"{train_data['normalized_energy']:.3f}",
         f"{test_data['normalized_energy']:.3f}",
         f"{prompts_data['normalized_energy']:.3f}"),
        ("Размер пересечения",
         f"{train_data['dim_intersection']}",
         f"{test_data['dim_intersection']}",
         f"{prompts_data['dim_intersection']}"),
        ("Энергия перекрытия",
         f"{train_data['overlap_energy']:.3f}",
         f"{test_data['overlap_energy']:.3f}",
         f"{prompts_data['overlap_energy']:.3f}"),
        ("Максимальное σ",
         f"{max(train_data['sigma']):.3f}",
         f"{max(test_data['sigma']):.3f}",
         f"{max(prompts_data['sigma']):.3f}"),
        ("Минимальный угол (°)",
         f"{np.degrees(min(train_data['angles_rad'])):.1f}",
         f"{np.degrees(min(test_data['angles_rad'])):.1f}",
         f"{np.degrees(min(prompts_data['angles_rad'])):.1f}")
    ]

    for metric in metrics_console:
        print(f"{metric[0]:<25} {metric[1]:<12} {metric[2]:<12} {metric[3]:<12}")

    print("=" * 80)
    print("Таблицы сохранены как 'summary_table.png' и 'detailed_metrics_table.png'")