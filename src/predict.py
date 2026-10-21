import joblib
import pandas as pd
from data_prep import load_data, clean_data, preprocess_data

def generar_predicciones(data_path, model_path):
    
    modelo = joblib.load(model_path)

    df_raw = load_data(data_path)
    df_clean = clean_data(df_raw.copy())
    df_processed = preprocess_data(df_clean)

    if 'Churn' in df_processed.columns:
        X_new = df_processed.drop(columns=['Churn'])
    else:
        X_new = df_processed

    predicciones = modelo.predict(X_new)
    probabilidades = modelo.predict_proba(X_new)[:, 1] 

    df_raw['Prediccion_Churn'] = predicciones
    df_raw['Probabilidad_Fuga (%)'] = (probabilidades * 100).round(2)

    return df_raw

if __name__ == "__main__":
    print("Iniciando motor de predicción...")
    
    archivo_datos = 'data/raw/Telco_Customer_Churn_Dataset.csv'
    archivo_modelo = 'models/modelo_churn_lr.pkl'
    
    resultados = generar_predicciones(archivo_datos, archivo_modelo)
    
    print("\nPredicciones generadas con éxito. Muestra de los primeros 5 clientes:")
    columnas_vista = ['gender', 'tenure', 'Contract', 'Prediccion_Churn', 'Probabilidad_Fuga (%)']
    print(resultados[columnas_vista].head())