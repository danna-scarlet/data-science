"""
Eres  un  analista  de  datos  en  un  centro  de  investigación  sobre  cambio  climático  y  
seguridad alimentaria. Te han encomendado evaluar cómo factores climáticos afectan la producción 
agrícola en  distintos  países.  Para  ello,  aplicarás  modelos  de  aprendizaje  supervisado  para
predecir  la producción de alimentos y clasificar los países según su vulnerabilidad.
1. Carga y exploración de datos (1 punto) 
• Carga el dataset proporcionado, que contiene información sobre temperatura media, 
cambio en las precipitaciones, frecuencia de sequías y producción agrícola en 
distintos países. 
• Analiza la distribución de las variables y detecta posibles valores atípicos o 
tendencias. 
2. Preprocesamiento y escalamiento de datos (2 puntos) 
• Aplica técnicas de normalización o estandarización a las variables numéricas. 
• Codifica correctamente cualquier variable categórica si fuera necesario. 
• Divide los datos en conjunto de entrenamiento y prueba (80%-20%). 
3. Aplicación de modelos de aprendizaje supervisado (4 puntos) 
• Regresión:  
o Entrena un modelo de regresión lineal para predecir la producción de 
alimentos. 
o Evalúa el modelo usando métricas como MAE, MSE y R2. 
o Compara con otros modelos de regresión (árbol de decisión, random forest). 
• Clasificación:  
o Crea una nueva variable categórica que clasifique los países en "Bajo", 
"Medio" y "Alto" impacto climático en la producción agrícola. 
o Entrena modelos de clasificación como K-Nearest Neighbors, Árbol de 
Decisión y Support Vector Machine. 
o Evalúa el desempeño usando matriz de confusión, precisión, sensibilidad y 
curva ROC-AUC. 
4. Optimización de modelos (2 puntos) 
• Ajusta hiperparámetros utilizando validación cruzada y búsqueda en grilla. 
• Aplica técnicas de regularización y analiza su impacto en los modelos. 
5. Análisis de resultados y conclusiones (1 punto) 
• Compara los modelos utilizados y justifica cuál ofrece mejores resultados para la 
predicción y clasificación. 
• Relaciona los hallazgos con posibles implicaciones en la seguridad alimentaria 
global.
"""

#1. Carga y exploración de datos (1 punto) 
#• Carga el dataset proporcionado, que contiene información sobre temperatura media, 
#cambio en las precipitaciones, frecuencia de sequías y producción agrícola en 
#distintos países. 
import pandas as pd

df = pd.read_csv("C:\\Users\\danna\\OneDrive\\Documentos\\GitHub\\data-science\\M6\\Evaluacion final\\cambio_climatico_agricultura.csv")
print("\n--------Carga y exploración de datos--------\n")
print(f'Dataset: \n{df.head()}\n')

#• Analiza la distribución de las variables y detecta posibles valores atípicos o 
#tendencias.
print(f'Descripción estadística: \n{df.describe()}\n')

#2. Preprocesamiento y escalamiento de datos (2 puntos) 
#• Aplica técnicas de normalización o estandarización a las variables numéricas. 
#• Codifica correctamente cualquier variable categórica si fuera necesario. 
#• Divide los datos en conjunto de entrenamiento y prueba (80%-20%). 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score

df['País'] = LabelEncoder().fit_transform(df['País'])

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

X = df_scaled.drop('Producción_alimentos', axis=1)
y = df_scaled['Producción_alimentos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3. Aplicación de modelos de aprendizaje supervisado (4 puntos) 
#• Regresión:  
#o Entrena un modelo de regresión lineal para predecir la producción de 
#alimentos. 
#o Evalúa el modelo usando métricas como MAE, MSE y R2. 
#o Compara con otros modelos de regresión (árbol de decisión, random forest). 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Modelo de regresión lineal
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n--------Aplicacion de modelos de aprendizaje supervisado--------\n")
print("\n1. Modelo de Regresión Lineal: \n")
print(f"MAE: {mean_absolute_error(y_test, y_pred_lr):.2f}\n")
print(f"MSE: {mean_squared_error(y_test, y_pred_lr):.2f}\n")
print(f"R2: {r2_score(y_test, y_pred_lr):.2f}\n")

# Modelo árbol de decisión
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n2. Árbol de Decisión:\n")
print(f"MAE: {mean_absolute_error(y_test, y_pred_dt):.2f}\n")
print(f"MSE: {mean_squared_error(y_test, y_pred_dt):.2f}\n")
print(f"R2: {r2_score(y_test, y_pred_dt):.2f}\n")

# Modelo random forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n3. Random Forest:\n")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}\n")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.2f}\n")
print(f"R2: {r2_score(y_test, y_pred_rf):.2f}\n")

#• Clasificación:  
#o Crea una nueva variable categórica que clasifique los países en "Bajo", 
#"Medio" y "Alto" impacto climático en la producción agrícola. 
#o Entrena modelos de clasificación como K-Nearest Neighbors, Árbol de 
#Decisión y Support Vector Machine. 
#o Evalúa el desempeño usando matriz de confusión, precisión, sensibilidad y 
#curva ROC-AUC.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Crear variable categórica de impacto climático
impact_bins = [-float('inf'), df['Producción_alimentos'].quantile(0.33), df['Producción_alimentos'].quantile(0.66), float('inf')]
impact_labels = ['Alto', 'Medio', 'Bajo']
df['Impacto_climatico'] = pd.cut(df['Producción_alimentos'], bins=impact_bins, labels=impact_labels)
print("\n--------Agregación de variable categórica de impacto climático-------\n")
print(f'Nueva variable categórica de impacto climático:\n{df["Impacto_climatico"].value_counts()}\n')

# Codificar variable objetivo
y_clf = LabelEncoder().fit_transform(df['Impacto_climatico'])
X_clf = df.drop(['Producción_alimentos', 'Impacto_climatico'], axis=1)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train_clf, y_train_clf)
y_pred_knn = knn.predict(X_test_clf)
print("1. Modelo de K-Nearest Neighbors (KNN):\n")
print("KNN Clasificación:\n", classification_report(y_test_clf, y_pred_knn, zero_division=0))
print("Matriz de confusión:\n", confusion_matrix(y_test_clf, y_pred_knn))

precision = precision_score(y_test_clf, y_pred_knn, average='weighted', zero_division=0)
recall = recall_score(y_test_clf, y_pred_knn, average='weighted', zero_division=0)
print(f'\nPrecisión: {precision:.2f}\n')
print(f'Sensibilidad (Recall): {recall:.2f}\n')

# Árbol de Decisión
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train_clf, y_train_clf)
y_pred_dtc = dtc.predict(X_test_clf)
print("2. Árbol de Decisión:\n")
print("Árbol de Decisión Clasificación:\n", classification_report(y_test_clf, y_pred_dtc, zero_division=0))
print("Matriz de confusión:\n", confusion_matrix(y_test_clf, y_pred_dtc))

precision = precision_score(y_test_clf, y_pred_dtc, average='weighted', zero_division=0)
recall = recall_score(y_test_clf, y_pred_dtc, average='weighted', zero_division=0)
print(f'\nPrecisión: {precision:.2f}\n')
print(f'Sensibilidad (Recall): {recall:.2f}\n')

# SVM
svc = SVC(probability=True, random_state=42)
svc.fit(X_train_clf, y_train_clf)
y_pred_svc = svc.predict(X_test_clf)
print("3. SVM:\n")
print("SVM Clasificación:\n", classification_report(y_test_clf, y_pred_svc, zero_division=0))
print("Matriz de confusión:\n", confusion_matrix(y_test_clf, y_pred_svc))

precision = precision_score(y_test_clf, y_pred_svc, average='weighted', zero_division=0)
recall = recall_score(y_test_clf, y_pred_svc, average='weighted', zero_division=0)
print(f'\nPrecisión: {precision:.2f}\n')
print(f'Sensibilidad (Recall): {recall:.2f}\n')

# ROC-AUC para clasificación multiclase (One-vs-Rest)
y_score_knn = knn.predict_proba(X_test_clf)
y_score_dtc = dtc.predict_proba(X_test_clf)
y_score_svc = svc.predict_proba(X_test_clf)

print("4. ROC-AUC Scores de los modelos aplicados:\n")
print("ROC-AUC KNN:", roc_auc_score(y_test_clf, y_score_knn, multi_class='ovr'))
print("ROC-AUC Árbol de Decisión:", roc_auc_score(y_test_clf, y_score_dtc, multi_class='ovr'))
print("ROC-AUC SVM:", roc_auc_score(y_test_clf, y_score_svc, multi_class='ovr'))