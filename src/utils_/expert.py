import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec

# Настройка стиля
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_and_analyze_data():
    """Загрузка и базовый анализ данных"""
    try:
        with open('../../predict.json', 'r') as f:
            data = json.load(f)

        analytic_scores = data['Analytic']['predict']
        finance_scores = data['Finance practical']['predict']

        return analytic_scores, finance_scores
    except FileNotFoundError:
        print("Файл predict.json не найден! Создаю тестовые данные...")
        # Создаем тестовые данные для демонстрации
        np.random.seed(42)
        analytic_scores = np.random.normal(0.67, 0.08, 50)
        finance_scores = np.random.normal(0.67, 0.10, 50)
        return analytic_scores, finance_scores


def create_comprehensive_plots():
    """Создание комплексных графиков анализа"""
    analytic_scores, finance_scores = load_and_analyze_data()

    # Создаем фигуру с несколькими субплогами
    fig = plt.figure(figsize=(20, 16))

    # Сетка для размещения графиков
    gs = GridSpec(3, 3, figure=fig)

    # 1. Гистограммы распределения оценок
    ax1 = fig.add_subplot(gs[0, 0])
    create_histogram_plot(ax1, analytic_scores, finance_scores)

    # 2. Boxplot сравнения распределений
    ax2 = fig.add_subplot(gs[0, 1])
    create_boxplot(ax2, analytic_scores, finance_scores)

    # 3. Кумулятивные распределения (CDF)
    ax3 = fig.add_subplot(gs[0, 2])
    create_cdf_plot(ax3, analytic_scores, finance_scores)

    # 4. Временные ряды оценок
    ax4 = fig.add_subplot(gs[1, :])
    create_timeseries_plot(ax4, analytic_scores, finance_scores)

    # 5. Scatter plot матриц признаков (первые 2 компоненты)
    ax5 = fig.add_subplot(gs[2, 0])
    create_scatter_plot(ax5, analytic_scores, finance_scores)

    # 6. Heatmap корреляций
    ax6 = fig.add_subplot(gs[2, 1])
    create_correlation_heatmap(ax6, analytic_scores, finance_scores)

    # 7. Violin plot
    ax7 = fig.add_subplot(gs[2, 2])
    create_violin_plot(ax7, analytic_scores, finance_scores)

    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Создаем отдельные графики высокого качества
    create_individual_plots(analytic_scores, finance_scores)


def create_histogram_plot(ax, analytic_scores, finance_scores):
    """Гистограмма распределения оценок"""
    bins = np.linspace(0.4, 1.0, 20)

    ax.hist(analytic_scores, bins=bins, alpha=0.7, label='Аналитик',
            color='blue', density=True)
    ax.hist(finance_scores, bins=bins, alpha=0.7, label='Финансист',
            color='red', density=True)

    ax.set_xlabel('Оценка качества')
    ax.set_ylabel('Плотность вероятности')
    ax.set_title('Распределение оценок качества ответов')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Добавляем линии средних значений
    ax.axvline(np.mean(analytic_scores), color='blue', linestyle='--',
               label=f'Среднее аналитик: {np.mean(analytic_scores):.3f}')
    ax.axvline(np.mean(finance_scores), color='red', linestyle='--',
               label=f'Среднее финансист: {np.mean(finance_scores):.3f}')
    ax.legend()


def create_boxplot(ax, analytic_scores, finance_scores):
    """Boxplot сравнения распределений"""
    data = [analytic_scores, finance_scores]
    labels = ['Аналитик', 'Финансист']

    box_plot = ax.boxplot(data, labels=labels, patch_artist=True)

    # Настройка цветов
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Оценка качества')
    ax.set_title('Сравнение распределений оценок\n(Boxplot)')
    ax.grid(True, alpha=0.3)

    # Добавляем аннотации с основными статистиками
    stats_text = (
        f'Аналитик:\n'
        f'Медиана: {np.median(analytic_scores):.3f}\n'
        f'IQR: {np.percentile(analytic_scores, 75) - np.percentile(analytic_scores, 25):.3f}\n\n'
        f'Финансист:\n'
        f'Медиана: {np.median(finance_scores):.3f}\n'
        f'IQR: {np.percentile(finance_scores, 75) - np.percentile(finance_scores, 25):.3f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)


def create_cdf_plot(ax, analytic_scores, finance_scores):
    """График кумулятивной функции распределения (CDF)"""

    def ecdf(data):
        """Эмпирическая CDF"""
        x = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        return x, y

    x_analytic, y_analytic = ecdf(analytic_scores)
    x_finance, y_finance = ecdf(finance_scores)

    ax.plot(x_analytic, y_analytic, label='Аналитик', linewidth=2, color='blue')
    ax.plot(x_finance, y_finance, label='Финансист', linewidth=2, color='red')

    ax.set_xlabel('Оценка качества')
    ax.set_ylabel('Кумулятивная вероятность')
    ax.set_title('Кумулятивное распределение оценок (CDF)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Добавляем линии для медиан
    median_analytic = np.median(analytic_scores)
    median_finance = np.median(finance_scores)
    ax.axvline(median_analytic, color='blue', linestyle='--', alpha=0.7)
    ax.axvline(median_finance, color='red', linestyle='--', alpha=0.7)


def create_timeseries_plot(ax, analytic_scores, finance_scores):
    """График временных рядов оценок"""
    messages = range(1, len(analytic_scores) + 1)

    ax.plot(messages, analytic_scores, label='Аналитик', linewidth=2, color='blue', marker='o', markersize=3)
    ax.plot(messages, finance_scores, label='Финансист', linewidth=2, color='red', marker='s', markersize=3)

    # Добавляем скользящее среднее с правильным выравниванием
    window = 5
    if len(analytic_scores) >= window:
        analytic_ma = np.convolve(analytic_scores, np.ones(window) / window, mode='valid')
        finance_ma = np.convolve(finance_scores, np.ones(window) / window, mode='valid')
        # Выравниваем по центру
        ma_messages = messages[window // 2:len(messages) - window // 2]

        # Проверяем размерности
        min_len = min(len(ma_messages), len(analytic_ma), len(finance_ma))
        ma_messages = ma_messages[:min_len]
        analytic_ma = analytic_ma[:min_len]
        finance_ma = finance_ma[:min_len]

        ax.plot(ma_messages, analytic_ma, label='Аналитик (скольз. среднее)',
                linewidth=3, color='darkblue', linestyle='--')
        ax.plot(ma_messages, finance_ma, label='Финансист (скольз. среднее)',
                linewidth=3, color='darkred', linestyle='--')

    ax.set_xlabel('Номер сообщения')
    ax.set_ylabel('Оценка качества')
    ax.set_title('Динамика оценок качества по ходу диалога')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Добавляем горизонтальные линии для средних
    ax.axhline(np.mean(analytic_scores), color='blue', linestyle=':', alpha=0.5)
    ax.axhline(np.mean(finance_scores), color='red', linestyle=':', alpha=0.5)


def create_scatter_plot(ax, analytic_scores, finance_scores):
    """Scatter plot оценок"""
    messages = range(1, len(analytic_scores) + 1)

    scatter1 = ax.scatter(messages, analytic_scores, c=analytic_scores,
                          cmap='Blues', alpha=0.7, s=50, label='Аналитик')
    scatter2 = ax.scatter(messages, finance_scores, c=finance_scores,
                          cmap='Reds', alpha=0.7, s=50, label='Финансист')

    ax.set_xlabel('Номер сообщения')
    ax.set_ylabel('Оценка качества')
    ax.set_title('Scatter plot оценок качества')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Добавляем colorbar
    plt.colorbar(scatter1, ax=ax, label='Оценка качества')


def create_correlation_heatmap(ax, analytic_scores, finance_scores):
    """Heatmap корреляций"""
    # Создаем матрицу корреляций
    correlation_matrix = np.corrcoef([analytic_scores, finance_scores])

    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Добавляем аннотации
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                    ha='center', va='center', fontsize=14,
                    color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Аналитик', 'Финансист'])
    ax.set_yticklabels(['Аналитик', 'Финансист'])
    ax.set_title('Матрица корреляций оценок')

    plt.colorbar(im, ax=ax, label='Коэффициент корреляции')


def create_violin_plot(ax, analytic_scores, finance_scores):
    """Violin plot распределений"""
    data = [analytic_scores, finance_scores]
    labels = ['Аналитик', 'Финансист']

    violin_parts = ax.violinplot(data, showmeans=True, showmedians=True)

    # Настройка цветов
    for pc, color in zip(violin_parts['bodies'], ['lightblue', 'lightcoral']):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    violin_parts['cbars'].set_color('black')
    violin_parts['cmins'].set_color('black')
    violin_parts['cmaxes'].set_color('black')
    violin_parts['cmeans'].set_color('red')
    violin_parts['cmedians'].set_color('blue')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Оценка качества')
    ax.set_title('Violin plot распределений оценок')
    ax.grid(True, alpha=0.3)


def create_individual_plots(analytic_scores, finance_scores):
    """Создание отдельных графиков высокого качества"""

    # 1. Детальный boxplot
    plt.figure(figsize=(10, 6))
    create_detailed_boxplot(analytic_scores, finance_scores)
    plt.savefig('detailed_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Статистическое сравнение
    plt.figure(figsize=(12, 8))
    create_statistical_comparison(analytic_scores, finance_scores)
    plt.savefig('statistical_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Распределение разностей
    plt.figure(figsize=(10, 6))
    create_difference_distribution(analytic_scores, finance_scores)
    plt.savefig('difference_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_detailed_boxplot(analytic_scores, finance_scores):
    """Детализированный boxplot с дополнительной статистикой"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Boxplot
    data = [analytic_scores, finance_scores]
    labels = ['Аналитик', 'Финансист']

    box_plot = ax1.boxplot(data, labels=labels, patch_artist=True, notch=True)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    ax1.set_ylabel('Оценка качества')
    ax1.set_title('Детализированное сравнение распределений')
    ax1.grid(True, alpha=0.3)

    # Статистическая сводка
    stats_data = {
        'Метрика': ['Среднее', 'Медиана', 'Стандартное отклонение',
                    'Минимум', 'Максимум', 'Количество'],
        'Аналитик': [
            f'{np.mean(analytic_scores):.3f}',
            f'{np.median(analytic_scores):.3f}',
            f'{np.std(analytic_scores):.3f}',
            f'{np.min(analytic_scores):.3f}',
            f'{np.max(analytic_scores):.3f}',
            f'{len(analytic_scores)}'
        ],
        'Финансист': [
            f'{np.mean(finance_scores):.3f}',
            f'{np.median(finance_scores):.3f}',
            f'{np.std(finance_scores):.3f}',
            f'{np.min(finance_scores):.3f}',
            f'{np.max(finance_scores):.3f}',
            f'{len(finance_scores)}'
        ]
    }

    # Создаем таблицу
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=list(zip(stats_data['Аналитик'], stats_data['Финансист'])),
                      rowLabels=stats_data['Метрика'],
                      colLabels=['Аналитик', 'Финансист'],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax2.set_title('Статистическая сводка')


def create_statistical_comparison(analytic_scores, finance_scores):
    """Статистическое сравнение распределений"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. QQ-plot для нормальности
    stats.probplot(analytic_scores, dist="norm", plot=ax1)
    ax1.set_title('QQ-plot: Аналитик (проверка нормальности)')

    stats.probplot(finance_scores, dist="norm", plot=ax2)
    ax2.set_title('QQ-plot: Финансист (проверка нормальности)')

    # 2. Плотность распределения с ядерной оценкой
    sns.kdeplot(analytic_scores, ax=ax3, label='Аналитик', fill=True, alpha=0.5)
    sns.kdeplot(finance_scores, ax=ax3, label='Финансист', fill=True, alpha=0.5)
    ax3.set_xlabel('Оценка качества')
    ax3.set_ylabel('Плотность вероятности')
    ax3.set_title('Ядерная оценка плотности распределения')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 3. Статистические тесты
    ax4.axis('off')

    # Выполняем статистические тесты
    t_stat, t_p = stats.ttest_ind(analytic_scores, finance_scores)
    ks_stat, ks_p = stats.ks_2samp(analytic_scores, finance_scores)
    mw_stat, mw_p = stats.mannwhitneyu(analytic_scores, finance_scores)

    test_results = [
        ['Тест', 'Статистика', 'p-value'],
        ['T-тест', f'{t_stat:.4f}', f'{t_p:.4f}'],
        ['Тест Колмогорова-Смирнова', f'{ks_stat:.4f}', f'{ks_p:.4f}'],
        ['U-тест Манна-Уитни', f'{mw_stat:.4f}', f'{mw_p:.4f}']
    ]

    table = ax4.table(cellText=test_results[1:],
                      rowLabels=[row[0] for row in test_results[1:]],
                      colLabels=test_results[0][1:],
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Результаты статистических тестов')


def create_difference_distribution(analytic_scores, finance_scores):
    """Распределение разностей между оценками"""
    # Вычисляем разности (аналитик - финансист)
    differences = np.array(analytic_scores) - np.array(finance_scores)

    plt.figure(figsize=(12, 8))

    # Гистограмма разностей
    plt.subplot(2, 2, 1)
    plt.hist(differences, bins=15, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Нулевая разность')
    plt.axvline(np.mean(differences), color='blue', linestyle='-',
                label=f'Средняя разность: {np.mean(differences):.3f}')
    plt.xlabel('Разность оценок (Аналитик - Финансист)')
    plt.ylabel('Частота')
    plt.title('Распределение разностей оценок')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Scatter plot разностей
    plt.subplot(2, 2, 2)
    messages = range(len(differences))
    colors = ['red' if diff < 0 else 'blue' for diff in differences]
    plt.scatter(messages, differences, c=colors, alpha=0.6)
    plt.axhline(0, color='black', linestyle='-', alpha=0.5)
    plt.axhline(np.mean(differences), color='blue', linestyle='--',
                label=f'Среднее: {np.mean(differences):.3f}')
    plt.xlabel('Номер сообщения')
    plt.ylabel('Разность оценок')
    plt.title('Разности оценок по сообщениям')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Кумулятивная разность
    plt.subplot(2, 2, 3)
    cumulative_diff = np.cumsum(differences)
    plt.plot(range(len(cumulative_diff)), cumulative_diff, linewidth=2, color='green')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Номер сообщения')
    plt.ylabel('Кумулятивная разность')
    plt.title('Накопленная разность оценок')
    plt.grid(True, alpha=0.3)

    # Процент сообщений, где аналитик лучше
    plt.subplot(2, 2, 4)
    analytic_better = np.sum(differences > 0)
    finance_better = np.sum(differences < 0)
    equal = np.sum(differences == 0)

    labels = ['Аналитик лучше', 'Финансист лучше', 'Равны']
    sizes = [analytic_better, finance_better, equal]
    colors = ['lightblue', 'lightcoral', 'lightgreen']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Сравнение качества ответов')

    plt.tight_layout()


# Запуск создания графиков
if __name__ == "__main__":
    print("Создание комплексных графиков анализа...")
    create_comprehensive_plots()

    print("Графики сохранены:")
    print("- comprehensive_analysis.png: комплексный анализ")
    print("- detailed_boxplot.png: детализированные boxplot")
    print("- statistical_comparison.png: статистическое сравнение")
    print("- difference_distribution.png: анализ разностей")