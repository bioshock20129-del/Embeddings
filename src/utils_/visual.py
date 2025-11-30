import json
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных из JSON файла
with open('../volumes.json', 'r') as f:
    data = json.load(f)

# Извлечение данных
train_data = data['train_text']
test_data = data['test_text']
prompts_data = data['prompts']

# 1. График нормализованной энергии
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

categories = ['Обучающие\nданные', 'Тестовые\nданные', 'Системные\nпромпты']
normalized_energy = [
    train_data['normalized_energy'],
    test_data['normalized_energy'],
    prompts_data['normalized_energy']
]

bars = ax1.bar(categories, normalized_energy, color=['#1f77b4', '#2ca02c', '#d62728'])
ax1.set_ylabel('Нормализованная энергия перекрытия')
ax1.set_title('Сравнение нормализованной энергии перекрытия подпространств', fontsize=14, fontweight='bold')

# Добавляем значения на столбцы
for bar, value in zip(bars, normalized_energy):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

# 2. График размеров пересечения
intersection_sizes = [
    train_data['dim_intersection'],
    test_data['dim_intersection'],
    prompts_data['dim_intersection']
]

bars = ax2.bar(categories, intersection_sizes, color=['#1f77b4', '#2ca02c', '#d62728'])
ax2.set_ylabel('Размер пересечения')
ax2.set_title('Сравнение размеров пересечения подпространств', fontsize=14, fontweight='bold')

# Добавляем значения на столбцы
for bar, value in zip(bars, intersection_sizes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('subspace_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. График сингулярных чисел
plt.figure(figsize=(12, 6))

# Берем первые 25 компонент для наглядности
components = range(1, 26)
train_sigma = train_data['sigma'][:25]
test_sigma = test_data['sigma'][:25]

plt.plot(components, train_sigma, 'o-', linewidth=2, markersize=6, label='Обучающие данные', color='#1f77b4')
plt.plot(components, test_sigma, 's-', linewidth=2, markersize=6, label='Тестовые данные', color='#2ca02c')

plt.xlabel('Номер компоненты')
plt.ylabel('Сингулярное число (σ)')
plt.title('Сингулярные числа для первых 25 компонент', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Добавляем горизонтальную линию для порога пересечения
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Полное пересечение (σ=1.0)')
plt.legend()

plt.tight_layout()
plt.savefig('singular_values.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. График канонических углов
plt.figure(figsize=(12, 6))

train_angles = np.degrees(train_data['angles_rad'][:25])  # Переводим в градусы
test_angles = np.degrees(test_data['angles_rad'][:25])

plt.plot(components, train_angles, 'o-', linewidth=2, markersize=6, label='Обучающие данные', color='#1f77b4')
plt.plot(components, test_angles, 's-', linewidth=2, markersize=6, label='Тестовые данные', color='#2ca02c')

plt.xlabel('Номер компоненты')
plt.ylabel('Канонический угол (градусы)')
plt.title('Канонические углы между подпространствами', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('canonical_angles.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Сводная таблица результатов
print("="*60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ АНАЛИЗА ПЕРЕСЕЧЕНИЯ ПОДПРОСТРАНСТВ")
print("="*60)
print(f"{'Метрика':<25} {'Обучающие':<12} {'Тестовые':<12} {'Промпты':<12}")
print("-"*60)
print(f"{'Норм. энергия':<25} {train_data['normalized_energy']:<12.3f} {test_data['normalized_energy']:<12.3f} {prompts_data['normalized_energy']:<12.3f}")
print(f"{'Размер пересечения':<25} {train_data['dim_intersection']:<12} {test_data['dim_intersection']:<12} {prompts_data['dim_intersection']:<12}")
print(f"{'Энергия перекрытия':<25} {train_data['overlap_energy']:<12.3f} {test_data['overlap_energy']:<12.3f} {prompts_data['overlap_energy']:<12.3f}")
print(f"{'Макс. σ':<25} {max(train_data['sigma']):<12.3f} {max(test_data['sigma']):<12.3f} {max(prompts_data['sigma']):<12.3f}")
print(f"{'Мин. угол (град)':<25} {np.degrees(min(train_data['angles_rad'])):<12.1f} {np.degrees(min(test_data['angles_rad'])):<12.1f} {np.degrees(min(prompts_data['angles_rad'])):<12.1f}")
print("="*60)