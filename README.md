# Predicción de Abandono de Clientes (Telco Customer Churn)

## 1. Descripción del problema de negocio
La alta tasa de cancelación (*churn*) genera pérdidas de ingresos recurrentes en la empresa de telecomunicaciones. El objetivo de este proyecto es identificar anticipadamente a los clientes propensos a abandonar el servicio para permitir la implementación de campañas de retención preventivas.

## 2. Objetivos del proyecto
Desarrollar, evaluar y comparar modelos de Machine Learning (supervisados y no supervisados) para predecir el *churn* y descubrir segmentos de clientes basados en su comportamiento de consumo y ciclo de vida.

## 3. Definición de KPIs
* **Tasa de Churn Global:** Línea base actual de cancelaciones para dimensionar el problema.
* **Recall (Clase 1):** Métrica técnica principal para minimizar los falsos negativos.
* **F1-Score:** Para evaluar el balance general de las predicciones dadas las clases desbalanceadas.

## 4. Fuentes de datos utilizadas
Conjunto de datos estructurado *Telco Customer Churn*, compuesto por 7.043 registros y 21 variables que incluyen datos demográficos, servicios contratados y comportamiento histórico de pago.

## 5. Metodología utilizada (CRISP-DM)
Se aplicó el estándar de la industria CRISP-DM, abarcando las fases de comprensión del negocio, comprensión y preparación de datos, modelamiento, evaluación de desempeño y análisis de impacto ético.

## 6. Preparación y análisis exploratorio de los datos (EDA)
* **Limpieza:** Se eliminó la variable identificadora `customerID`, se transformó `TotalCharges` a formato numérico y se imputaron los valores nulos con 0.
* **Transformación:** Se binarizó la variable objetivo y se aplicó *One-Hot Encoding* a las variables categóricas.
* **Hallazgos EDA:** Los contratos mensuales (*Month-to-month*) y el método de pago por cheque electrónico presentan las mayores concentraciones de abandono.

## 7. Implementación de Modelos
* **Supervisados (Clasificación):** Se entrenaron y compararon algoritmos de Regresión Logística y Random Forest.
* **No Supervisado (Agrupación):** Se implementó K-Means (k=3) sobre las variables `tenure` (antigüedad) y `MonthlyCharges` (cargo mensual) para segmentar a los clientes en grupos de valor y riesgo.

## 8. Evaluación de Resultados
La Regresión Logística superó al Random Forest en la métrica clave de *Recall* para la clase positiva, demostrando mayor eficacia para detectar efectivamente a los clientes en riesgo de fuga.

## 9. Impacto Ético y Privacidad
Los datos fueron anonimizados eliminando identificadores personales. Se asume un riesgo de falsos positivos, lo cual representa un costo de marketing aceptable. El modelo debe auditarse periódicamente para evitar sesgos discriminatorios basados en edad (`SeniorCitizen`) o género.