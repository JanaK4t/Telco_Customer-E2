import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from data_prep import load_data, clean_data, preprocess_data, get_train_test_data

def main():
    print("Cargando y preprocesando el dataset...")
    df = load_data('data/raw/Telco_Customer_Churn_Dataset.csv')
    df_processed = preprocess_data(clean_data(df))
    
    # Aprendizaje No Supervisado: K-Means
    print("\nEntrenando Modelo No Supervisado (K-Means)...")
    X_cluster = df_processed[['tenure', 'MonthlyCharges']]
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    
    df_processed['Cluster'] = modelo_kmeans.fit_predict(X_cluster)
    print("Centroides de los segmentos encontrados (Antigüedad, Cargo Mensual):")
    print(modelo_kmeans.cluster_centers_)

    df_processed.to_csv('data/processed/Telco_Customer_Churn_Processed.csv', index=False)

    print("\nSeparando datos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = get_train_test_data(df_processed)

    print("Escalando las variables numéricas...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Entrenar Modelos Supervisados
    print("\nEntrenando Modelo 1: Regresión Logística...")
    modelo_lr = LogisticRegression(max_iter=1000, random_state=42)
    modelo_lr.fit(X_train, y_train)

    print("Entrenando Modelo 2: Random Forest...")
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    modelo_rf.fit(X_train, y_train)

    # 3. Evaluación
    print("\n--- Resultados Regresión Logística ---")
    y_pred_lr = modelo_lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr))
    
    print("\n--- Resultados Random Forest ---")
    y_pred_rf = modelo_rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf))

    print("\nExportando modelos a la carpeta 'models'...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(modelo_lr, 'models/modelo_churn_lr.pkl')
    joblib.dump(modelo_rf, 'models/modelo_churn_rf.pkl')
    joblib.dump(modelo_kmeans, 'models/modelo_churn_kmeans.pkl')
    print("¡Entrenamiento y exportación finalizados!")

if __name__ == "__main__":
    main()