import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    print("Iniciando motor de predicción...")

    df_processed = pd.read_csv('data/processed/Telco_Customer_Churn_Processed.csv')
    df_raw = pd.read_csv('data/raw/Telco_Customer_Churn_Dataset.csv')

    X_full = df_processed.drop(columns=['Churn'])
    muestra_original = df_raw.head(5)[['gender', 'tenure', 'Contract']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_muestra = X_scaled[:5]

    # Cargar el modelo ganador (Regresión Logística)
    modelo_lr = joblib.load('models/modelo_churn_lr.pkl')

    predicciones = modelo_lr.predict(X_muestra)
    probabilidades = modelo_lr.predict_proba(X_muestra)[:, 1] * 100

    resultados = muestra_original.copy()
    resultados['Prediccion_Churn'] = predicciones
    resultados['Probabilidad_Fuga (%)'] = probabilidades.round(2)

    print("\nPredicciones generadas con éxito. Muestra de los primeros 5 clientes:")
    print(resultados)

if __name__ == "__main__":
    main()