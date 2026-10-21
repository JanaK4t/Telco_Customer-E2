import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    return df

def preprocess_data(df):
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    
    # Aplicar One-Hot Encoding al resto de variables categóricas
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded

def get_train_test_data(df, target_col='Churn', test_size=0.2, random_state=42):
 
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test