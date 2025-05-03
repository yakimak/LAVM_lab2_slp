import pandas as pd

df = pd.read_csv('credit_data_1m.csv')

print("=== ОСНОВНАЯ ИНФОРМАЦИЯ ===")
print(f"Общее количество записей: {len(df):,}")
print(f"Количество признаков: {len(df.columns)}")
print(f"Доля дефолтов (Default=1): {df['Default'].mean():.2%}")
print(f"Доля недефолтов (Default=0): {1 - df['Default'].mean():.2%}\n")

print("=== ОСНОВНЫЕ СТАТИСТИКИ ЧИСЛОВЫХ ПРИЗНАКОВ ===")
numeric_stats = df.describe(percentiles=[.25, .75, .99]).transpose()
print(numeric_stats.round(2))

print("\n=== КОРРЕЛЯЦИЯ С TARGET (Default) ===")
corr_with_target = df.corr()['Default'].sort_values(ascending=False)
print(corr_with_target.round(4))

binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'Default']
print("\n=== РАСПРЕДЕЛЕНИЕ БИНАРНЫХ ПРИЗНАКОВ ===")
binary_stats = pd.DataFrame()
for col in binary_cols:
    binary_stats[col] = df[col].value_counts(normalize=True).round(4) * 100
print(binary_stats.transpose())
