import numpy as np
import pandas as pd

np.random.seed(42)
m = 1_000_000

def generate_one_hot(options, size):
    choices = np.random.choice(options, size=size)
    return pd.get_dummies(choices, prefix='', prefix_sep='').astype(int)

data = {
    'Age': np.random.normal(loc=45, scale=15, size=m).clip(18, 100),
    **generate_one_hot(['Male', 'Female'], m),
    **generate_one_hot(['Single', 'Married', 'Divorced'], m),
    'Dependents': np.random.poisson(lam=1, size=m).clip(0, 5),
    
    'Income': np.random.lognormal(mean=3, sigma=0.7, size=m),
    'Debt': np.random.exponential(scale=0.6, size=m),
    'Delinquency_days': np.random.exponential(scale=60, size=m),
    'Convictions': np.random.binomial(1, 0.15, size=m),
    'Job_changes': np.random.poisson(lam=3, size=m),
    
    'Capital': np.random.lognormal(mean=4, sigma=1, size=m),
    'City_population': np.random.lognormal(mean=5, sigma=2, size=m),
    'Active_loans': np.random.poisson(lam=4, size=m),

    'Collateral': np.random.binomial(1, 0.3, size=m),
    'Guarantors': np.random.binomial(1, 0.2, size=m),
    'Closed_loans': np.random.poisson(lam=5, size=m),
    'Loan_amount': np.random.lognormal(mean=5, sigma=0.8, size=m),
    'Interest_rate': np.random.normal(loc=15, scale=5, size=m).clip(5, 30),
    'Job_experience': np.random.normal(loc=7, scale=4, size=m).clip(0, 15),
    'Lawsuits': np.random.poisson(lam=0.5, size=m),
    'Travels_abroad': np.random.poisson(lam=1, size=m),
    'GTO_badge': np.random.binomial(1, 0.3, size=m)


}

education = np.random.choice(['School', 'College', 'University'], size=m, p=[0.4, 0.4, 0.2])
data.update(pd.get_dummies(education, prefix='Edu').astype(int))

loan_type = np.random.choice(['Consumer', 'Car', 'Mortgage'], size=m, p=[0.6, 0.3, 0.1])
data.update(pd.get_dummies(loan_type, prefix='Loan').astype(int))

df = pd.DataFrame(data)

numeric_cols = [
    'Age', 'Dependents', 'Income', 'Debt', 'Delinquency_days',
    'Job_changes', 'Capital', 'City_population', 'Active_loans'
]
df[numeric_cols] = df[numeric_cols].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

def determine_default(row):
    debt_ratio = row['Debt'] / (row['Income'] + 1e-6)
    
    conditions = [
        (debt_ratio > 0.5) & (row['Delinquency_days'] > 0.4),
        (debt_ratio > 0.5) & (row['Convictions'] > 0.5),
        (debt_ratio > 0.5) & (row['Job_changes'] > 0.4),
        (row['Delinquency_days'] > 0.5),
        (row['Debt']> 0.5)
    ]
    
    return 1 if any(conditions) else 0

df['Default'] = df.apply(determine_default, axis=1)

noise = np.random.rand(m) < 0.01
df.loc[noise, 'Default'] = 1 - df.loc[noise, 'Default']

print(f"Доля дефолтов: {df['Default'].mean():.2%}")

df.to_csv('credit_data_1m.csv', index=False)