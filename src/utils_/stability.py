def analyze_stability_patterns(analytic_scores, finance_scores):
    """Анализ паттернов стабильности"""

    # Коэффициент вариации
    cv_analytic = np.std(analytic_scores) / np.mean(analytic_scores)
    cv_finance = np.std(finance_scores) / np.mean(finance_scores)

    # Процент сообщений в пределах одного стандартного отклонения от среднего
    within_1std_analytic = np.mean(
        (analytic_scores >= np.mean(analytic_scores) - np.std(analytic_scores)) &
        (analytic_scores <= np.mean(analytic_scores) + np.std(analytic_scores))
    ) * 100

    within_1std_finance = np.mean(
        (finance_scores >= np.mean(finance_scores) - np.std(finance_scores)) &
        (finance_scores <= np.mean(finance_scores) + np.std(finance_scores))
    ) * 100

    print("\n" + "=" * 50)
    print("АНАЛИЗ СТАБИЛЬНОСТИ ВЫПОЛНЕНИЯ")
    print("=" * 50)
    print(f"Коэффициент вариации:")
    print(f"  Аналитик: {cv_analytic:.3f} (более стабильный)")
    print(f"  Финансист: {cv_finance:.3f} (менее стабильный)")

    print(f"\nСообщения в пределах ±1σ от среднего:")
    print(f"  Аналитик: {within_1std_analytic:.1f}%")
    print(f"  Финансист: {within_1std_finance:.1f}%")

    # Анализ последовательностей
    def count_consistency_changes(scores, threshold=0.05):
        changes = 0
        for i in range(1, len(scores)):
            if abs(scores[i] - scores[i - 1]) > threshold:
                changes += 1
        return changes

    analytic_changes = count_consistency_changes(analytic_scores)
    finance_changes = count_consistency_changes(finance_scores)

    print(f"\nРезкие изменения качества (>0.05 между сообщениями):")
    print(f"  Аналитик: {analytic_changes} из {len(analytic_scores) - 1}")
    print(f"  Финансист: {finance_changes} из {len(finance_scores) - 1}")

# Запустите эту функцию с вашими данными