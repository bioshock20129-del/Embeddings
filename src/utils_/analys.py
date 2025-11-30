import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
import pandas as pd


class PredictionAnalyzer:
    def __init__(self, file_path='../../predict.json'):
        self.file_path = file_path
        self.data = None
        self.analytic_scores = None
        self.finance_scores = None
        self.analytic_matrix = None
        self.finance_matrix = None

    def load_data(self):
        """Загрузка данных из JSON файла"""
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)

        self.analytic_scores = np.array(self.data['Analytic']['predict'])
        self.finance_scores = np.array(self.data['Finance practical']['predict'])
        self.analytic_matrix = np.array(self.data['Analytic']['matrix'])
        self.finance_matrix = np.array(self.data['Finance practical']['matrix'])

        print("Данные успешно загружены")
        print(f"Аналитик: {len(self.analytic_scores)} сообщений")
        print(f"Финансист: {len(self.finance_scores)} сообщений")

    def basic_statistics(self):
        """Базовая статистика оценок"""
        print("\n" + "=" * 50)
        print("БАЗОВАЯ СТАТИСТИКА ОЦЕНОК")
        print("=" * 50)

        stats_data = {
            'Метрика': ['Среднее', 'Медиана', 'Стандартное отклонение',
                        'Минимум', 'Максимум', 'Количество'],
            'Аналитик': [
                f"{np.mean(self.analytic_scores):.3f}",
                f"{np.median(self.analytic_scores):.3f}",
                f"{np.std(self.analytic_scores):.3f}",
                f"{np.min(self.analytic_scores):.3f}",
                f"{np.max(self.analytic_scores):.3f}",
                f"{len(self.analytic_scores)}"
            ],
            'Финансист': [
                f"{np.mean(self.finance_scores):.3f}",
                f"{np.median(self.finance_scores):.3f}",
                f"{np.std(self.finance_scores):.3f}",
                f"{np.min(self.finance_scores):.3f}",
                f"{np.max(self.finance_scores):.3f}",
                f"{len(self.finance_scores)}"
            ]
        }

        df_stats = pd.DataFrame(stats_data)
        print(df_stats.to_string(index=False))

        # Статистические тесты
        print("\nСТАТИСТИЧЕСКИЕ ТЕСТЫ:")
        t_stat, t_p = stats.ttest_ind(self.analytic_scores, self.finance_scores)
        ks_stat, ks_p = stats.ks_2samp(self.analytic_scores, self.finance_scores)
        mw_stat, mw_p = stats.mannwhitneyu(self.analytic_scores, self.finance_scores)

        print(f"T-тест: t={t_stat:.4f}, p={t_p:.4f}")
        print(f"Тест Колмогорова-Смирнова: D={ks_stat:.4f}, p={ks_p:.4f}")
        print(f"U-тест Манна-Уитни: U={mw_stat:.4f}, p={mw_p:.4f}")

        if t_p < 0.05:
            print("✓ Различия статистически значимы (p < 0.05)")
        else:
            print("× Различия не статистически значимы")

    def plot_distributions(self):
        """Визуализация распределений оценок"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Анализ распределений оценок качества ответов', fontsize=16, fontweight='bold')

        # 1. Гистограммы
        axes[0, 0].hist(self.analytic_scores, bins=15, alpha=0.7, color='blue',
                        label='Аналитик', density=True)
        axes[0, 0].hist(self.finance_scores, bins=15, alpha=0.7, color='red',
                        label='Финансист', density=True)
        axes[0, 0].set_xlabel('Оценка качества')
        axes[0, 0].set_ylabel('Плотность вероятности')
        axes[0, 0].set_title('Гистограмма распределений')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Boxplot
        box_data = [self.analytic_scores, self.finance_scores]
        box_labels = ['Аналитик', 'Финансист']
        bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[0, 1].set_ylabel('Оценка качества')
        axes[0, 1].set_title('Boxplot распределений')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Violin plot
        violin_parts = axes[0, 2].violinplot(box_data, showmeans=True, showmedians=True)
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        axes[0, 2].set_xticks([1, 2])
        axes[0, 2].set_xticklabels(box_labels)
        axes[0, 2].set_ylabel('Оценка качества')
        axes[0, 2].set_title('Violin plot')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Временные ряды
        messages = range(1, len(self.analytic_scores) + 1)
        axes[1, 0].plot(messages, self.analytic_scores, 'o-', label='Аналитик',
                        color='blue', markersize=4)
        axes[1, 0].plot(messages, self.finance_scores, 's-', label='Финансист',
                        color='red', markersize=4)
        axes[1, 0].set_xlabel('Номер сообщения')
        axes[1, 0].set_ylabel('Оценка качества')
        axes[1, 0].set_title('Динамика оценок по ходу диалога')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Кумулятивное распределение (CDF)
        def ecdf(data):
            x = np.sort(data)
            y = np.arange(1, len(data) + 1) / len(data)
            return x, y

        x_analytic, y_analytic = ecdf(self.analytic_scores)
        x_finance, y_finance = ecdf(self.finance_scores)
        axes[1, 1].plot(x_analytic, y_analytic, label='Аналитик', linewidth=2)
        axes[1, 1].plot(x_finance, y_finance, label='Финансист', linewidth=2)
        axes[1, 1].set_xlabel('Оценка качества')
        axes[1, 1].set_ylabel('Кумулятивная вероятность')
        axes[1, 1].set_title('Кумулятивная функция распределения (CDF)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Scatter plot
        scatter = axes[1, 2].scatter(messages, self.analytic_scores,
                                     c=self.analytic_scores, cmap='Blues',
                                     alpha=0.7, s=50, label='Аналитик')
        axes[1, 2].scatter(messages, self.finance_scores,
                           c=self.finance_scores, cmap='Reds',
                           alpha=0.7, s=50, label='Финансист')
        axes[1, 2].set_xlabel('Номер сообщения')
        axes[1, 2].set_ylabel('Оценка качества')
        axes[1, 2].set_title('Scatter plot оценок')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('predictions_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_score_categories(self):
        """Анализ категорий оценок"""
        print("\n" + "=" * 50)
        print("АНАЛИЗ КАТЕГОРИЙ ОЦЕНОК")
        print("=" * 50)

        # Определяем категории качества
        bins = [0, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ['Низкое (<0.6)', 'Среднее (0.6-0.7)', 'Хорошее (0.7-0.8)',
                  'Высокое (0.8-0.9)', 'Отличное (>0.9)']

        analytic_categories = pd.cut(self.analytic_scores, bins=bins, labels=labels)
        finance_categories = pd.cut(self.finance_scores, bins=bins, labels=labels)

        analytic_counts = pd.value_counts(analytic_categories, sort=False)
        finance_counts = pd.value_counts(finance_categories, sort=False)

        category_df = pd.DataFrame({
            'Категория': labels,
            'Аналитик': analytic_counts.values,
            'Финансист': finance_counts.values,
            'Аналитик %': (analytic_counts.values / len(self.analytic_scores) * 100).round(1),
            'Финансист %': (finance_counts.values / len(self.finance_scores) * 100).round(1)
        })

        print(category_df.to_string(index=False))

        # Визуализация категорий
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Столбчатая диаграмма
        x = np.arange(len(labels))
        width = 0.35
        ax1.bar(x - width / 2, analytic_counts.values, width, label='Аналитик', alpha=0.7)
        ax1.bar(x + width / 2, finance_counts.values, width, label='Финансист', alpha=0.7)
        ax1.set_xlabel('Категория качества')
        ax1.set_ylabel('Количество сообщений')
        ax1.set_title('Распределение по категориям качества')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Круговая диаграмма для аналитика
        ax2.pie(analytic_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Аналитик: распределение по категориям')

        plt.tight_layout()
        plt.savefig('score_categories_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_matrix_structure(self):
        """Анализ структуры матриц признаков"""
        print("\n" + "=" * 50)
        print("АНАЛИЗ СТРУКТУРЫ МАТРИЦ ПРИЗНАКОВ")
        print("=" * 50)

        # Основные характеристики матриц
        print(f"Размерность матрицы аналитика: {self.analytic_matrix.shape}")
        print(f"Размерность матрицы финансиста: {self.finance_matrix.shape}")

        # Анализ вариации по компонентам
        analytic_var = np.var(self.analytic_matrix, axis=0)
        finance_var = np.var(self.finance_matrix, axis=0)

        print(f"\nДисперсия по компонентам (первые 10):")
        print(f"Аналитик: {analytic_var[:10]}")
        print(f"Финансист: {finance_var[:10]}")

        # Визуализация дисперсии
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Дисперсия по компонентам
        components = range(1, min(20, len(analytic_var)) + 1)
        axes[0, 0].plot(components, analytic_var[:len(components)], 'o-',
                        label='Аналитик', color='blue')
        axes[0, 0].plot(components, finance_var[:len(components)], 's-',
                        label='Финансист', color='red')
        axes[0, 0].set_xlabel('Номер компоненты')
        axes[0, 0].set_ylabel('Дисперсия')
        axes[0, 0].set_title('Дисперсия по компонентам SVD')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. PCA визуализация
        pca = PCA(n_components=2)
        combined_matrix = np.vstack([self.analytic_matrix, self.finance_matrix])
        pca_result = pca.fit_transform(combined_matrix)

        analytic_2d = pca_result[:len(self.analytic_matrix)]
        finance_2d = pca_result[len(self.analytic_matrix):]

        axes[0, 1].scatter(analytic_2d[:, 0], analytic_2d[:, 1], alpha=0.6,
                           label='Аналитик', color='blue')
        axes[0, 1].scatter(finance_2d[:, 0], finance_2d[:, 1], alpha=0.6,
                           label='Финансист', color='red')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0, 1].set_title('PCA проекция матриц признаков')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Тепловая карта корреляций между оценками и компонентами
        # Для аналитика
        analytic_correlations = []
        for i in range(min(10, self.analytic_matrix.shape[1])):
            corr = np.corrcoef(self.analytic_scores, self.analytic_matrix[:, i])[0, 1]
            analytic_correlations.append(corr)

        finance_correlations = []
        for i in range(min(10, self.finance_matrix.shape[1])):
            corr = np.corrcoef(self.finance_scores, self.finance_matrix[:, i])[0, 1]
            finance_correlations.append(corr)

        correlation_matrix = np.array([analytic_correlations, finance_correlations])
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_xticks(range(len(analytic_correlations)))
        axes[1, 0].set_xticklabels([f'C{i + 1}' for i in range(len(analytic_correlations))])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_yticklabels(['Аналитик', 'Финансист'])
        axes[1, 0].set_title('Корреляция: оценки ↔ компоненты')
        plt.colorbar(im, ax=axes[1, 0])

        # 4. Распределение значений в матрицах
        axes[1, 1].hist(self.analytic_matrix.flatten(), bins=50, alpha=0.7,
                        label='Аналитик', color='blue', density=True)
        axes[1, 1].hist(self.finance_matrix.flatten(), bins=50, alpha=0.7,
                        label='Финансист', color='red', density=True)
        axes[1, 1].set_xlabel('Значение компоненты')
        axes[1, 1].set_ylabel('Плотность')
        axes[1, 1].set_title('Распределение значений в матрицах')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('matrix_structure_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"\nОбъясненная дисперсия PCA: {pca.explained_variance_ratio_}")
        print(f"Суммарная объясненная дисперсия: {np.sum(pca.explained_variance_ratio_):.2%}")

    def run_complete_analysis(self):
        """Запуск полного анализа"""
        print("НАЧАЛО АНАЛИЗА ДАННЫХ ИЗ PREDICT.JSON")
        print("=" * 60)

        self.load_data()
        self.basic_statistics()
        self.plot_distributions()
        self.analyze_score_categories()
        self.analyze_matrix_structure()

        print("\n" + "=" * 60)
        print("АНАЛИЗ ЗАВЕРШЕН")
        print("Созданные файлы:")
        print("- predictions_distribution_analysis.png")
        print("- score_categories_analysis.png")
        print("- matrix_structure_analysis.png")


# Запуск анализа
if __name__ == "__main__":
    analyzer = PredictionAnalyzer()
    analyzer.run_complete_analysis()
