import pandas as pd
from sklearn.datasets import fetch_openml

# Загрузка датасета
data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
df = data.frame

# Сохранение в CSV
df.to_csv('workers_compensation.csv', index=False)

print("Датасет успешно сохранён как 'workers_compensation.csv'")
print(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")