# Predicción de Abandono de Clientes (Telco Customer Churn)

## 1. Descripción del problema de negocio
La alta tasa de cancelación (*churn*) genera pérdidas de ingresos recurrentes en la empresa de telecomunicaciones. El objetivo de este proyecto es identificar anticipadamente a los clientes propensos a abandonar el servicio para permitir la implementación de campañas de retención preventivas.

## 2. Objetivos del proyecto
Desarrollar, evaluar y comparar modelos de Machine Learning (supervisados y no supervisados) para predecir el *churn* y descubrir segmentos de clientes basados en su comportamiento de consumo y ciclo de vida.

## 3. Definición de KPIs
* **Tasa de Churn Global:** Línea base actual de cancelaciones para dimensionar el problema.
* **Recall (Clase 1):** Métrica técnica principal para minimizar los falsos negativos (clientes que se van sin ser detectados por el modelo).
* **F1-Score:** Para evaluar el balance general de las predicciones dadas las clases desbalanceadas.

## 4. Fuentes de datos utilizadas
Conjunto de datos estructurado *Telco Customer Churn*, compuesto por 7.043 registros y 21 variables que incluyen datos demográficos, servicios contratados y comportamiento histórico de pago.

## 5. Metodología utilizada (CRISP-DM)
Se aplicó el estándar de la industria CRISP-DM, abarcando las fases de comprensión del negocio, comprensión y preparación de datos, modelamiento, evaluación de desempeño y análisis de impacto ético.

## 6. Preparación e Ingeniería de Características
* **Limpieza:** Se eliminó la variable `customerID`, se transformó `TotalCharges` a numérico y se imputaron nulos con 0.
* **Transformación:** Binarización de la variable objetivo, aplicación de *One-Hot Encoding* a variables categóricas y escalado de variables numéricas mediante `StandardScaler`.
* **Feature Engineering:** Se inyectó la etiqueta de segmento de cliente (generada por K-Means) como una nueva variable predictiva para enriquecer los modelos supervisados.

## 7. Implementación de Modelos
* **No Supervisado (K-Means):** Segmentación en 3 clústeres utilizando `tenure` y `MonthlyCharges` (Perfiles: Valor Básico, Alto Riesgo y Premium/Leales).
* **Supervisados (Clasificación):** Entrenamiento y comparación directa entre Regresión Logística y Random Forest.

## 8. Evaluación de Resultados
La Regresión Logística fue seleccionada como el modelo definitivo al lograr un *Recall* del 56% para la clase positiva, superando el 48% alcanzado por el Random Forest. El algoritmo lineal demostró mayor capacidad para minimizar los falsos negativos tras la estandarización de datos y el enriquecimiento del dataset con los clústeres.

## 9. Impacto Ético y Privacidad
Los datos fueron anonimizados eliminando identificadores personales. Se asume un riesgo de falsos positivos (ofrecer descuentos a clientes que no se iban a ir), lo cual representa un costo de marketing aceptable. El modelo debe auditarse periódicamente para evitar sesgos discriminatorios.
