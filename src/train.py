import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from data_prep import load_data, clean_data, preprocess_data, get_train_test_data

def main():
    print("Cargando el dataset...")
    df = load_data('data/raw/Telco_Customer_Churn_Dataset.csv')

    
    print("Aplicando limpieza y One-Hot Encoding...")
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)

    

    
    print("Guardando datos limpios en data/processed/...")
    df_processed.to_csv('data/processed/Telco_Customer_Churn_Processed.csv', index=False)



    print("Separando datos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = get_train_test_data(df_processed)

    print("Entrenando Regresión Logística...")
    modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
    modelo_lr.fit(X_train, y_train)

    print("\n--- Resultados del Modelo ---")
    y_pred = modelo_lr.predict(X_test)
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    print("\nExportando modelo a la carpeta 'models/'...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo_lr, 'models/modelo_churn_lr.pkl')
    print("¡Entrenamiento finalizado con éxito!")

if __name__ == "__main__":
    main()