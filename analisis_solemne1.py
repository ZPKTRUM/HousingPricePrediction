# en la terminal antes de ejecutar este script, instalar las librerías necesarias con:
# pip install pandas numpy matplotlib seaborn scipy scikit-learn missingno

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.decomposition import PCA
import missingno as msno
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

# Cargar datos
df = pd.read_csv('Housing_price_prediction.csv')

# 1. Descripción del conjunto de datos
print("1. Descripción del conjunto de datos:")
print(f"Dimensiones: {df.shape[0]} observaciones, {df.shape[1]} variables")
print("\nTipos de variables:")
print(df.dtypes)
print("\nPrimeras 5 filas:")
print(df.head())

# 2. Análisis gráfico de variables categóricas (Gráficos de barras)
categorical_cols = ['mainroad', 'guestroom', 'basement',
                    'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

plt.figure(figsize=(8, 8))
for i, col in enumerate(categorical_cols, 1):
    ax = plt.subplot(3, 3, i)
    counts = df[col].value_counts()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(col)
plt.tight_layout()
plt.show()


# 3. Tabla resumen de variables numéricas
numeric_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
print("3. Medidas estadísticas de variables numéricas:")
print(df[numeric_cols].describe())

# 4. Histograma con curva de densidad para 'price'
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True, bins=30)
plt.title('Distribución de Precios de Viviendas')
plt.show()

# 5. Análisis de normalidad para 'price' (QQ plot y Shapiro-Wilk)
plt.figure(figsize=(10, 6))
stats.probplot(df['price'], dist="norm", plot=plt)
plt.title('Q-Q Plot para Precio')
plt.show()

stat, p_value = stats.shapiro(df['price'])
print(f"Shapiro-Wilk Test: estadístico={stat}, p-value={p_value}")

# 6. Identificación de datos atípicos (boxplot)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'Boxplot de {col}')

plt.tight_layout()
plt.show()

# 7. Relación entre variable numérica y categórica (price vs furnishingstatus)
plt.figure(figsize=(10, 6))
sns.boxplot(x='furnishingstatus', y='price', data=df)
plt.title('Precio por Estado de Amueblado')
plt.show()

# 8. Análisis de datos faltantes
plt.figure(figsize=(10, 6))
msno.bar(df)
plt.title('Datos Faltantes por Columna')
plt.show()

# 9. Matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación')
plt.show()

# 10. Análisis de Componentes Principales (ACP)
# Preprocesamiento para ACP
le = LabelEncoder()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Varianza explicada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada por Componentes')
plt.show()

# Contribución de variables a las dos primeras componentes
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
plt.figure(figsize=(10, 6))
plt.scatter(loadings[:, 0], loadings[:, 1])
for i, col in enumerate(df_encoded.columns):
    plt.annotate(col, (loadings[i, 0], loadings[i, 1]))
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('Cargas de Variables en ACP')
plt.show()

# 11. Modelo 1: Regresión logística con todas las variables
# Preparar datos para el modelo
X = df_encoded.drop('furnishingstatus', axis=1)
y = df_encoded['furnishingstatus']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model1 = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=1000)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

print("11. Modelo 1 - Todas las variables:")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred1))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred1))

# 12. Modelo 2: Método backward (eliminación recursiva de variables)
# Implementación manual backward con importancia de características
selector = RFE(LogisticRegression(multi_class='multinomial',
               solver='lbfgs', max_iter=1000), n_features_to_select=5)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

model2 = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=1000)
model2.fit(X_train_selected, y_train)
y_pred2 = model2.predict(X_test_selected)

print("12. Modelo 2 - Backward Selection:")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred2))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred2))

# 13. Comparación de modelos
metrics = {
    'Modelo': ['Todas las variables', 'Backward Selection'],
    'Accuracy': [accuracy_score(y_test, y_pred1), accuracy_score(y_test, y_pred2)],
    'Precision': [precision_score(y_test, y_pred1, average='weighted'), precision_score(y_test, y_pred2, average='weighted')],
    'Recall': [recall_score(y_test, y_pred1, average='weighted'), recall_score(y_test, y_pred2, average='weighted')],
    'F1-Score': [f1_score(y_test, y_pred1, average='weighted'), f1_score(y_test, y_pred2, average='weighted')]
}
metrics_df = pd.DataFrame(metrics)
print("13. Comparación de Modelos:")
print(metrics_df)

# 14. Curvas ROC para modelos (multiclase)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

# Curva ROC para Modelo 1
fpr1, tpr1, roc_auc1 = dict(), dict(), dict()
for i in range(n_classes):
    fpr1[i], tpr1[i], _ = roc_curve(
        y_test_bin[:, i], model1.predict_proba(X_test)[:, i])
    roc_auc1[i] = auc(fpr1[i], tpr1[i])

# Curva ROC para Modelo 2
fpr2, tpr2, roc_auc2 = dict(), dict(), dict()
for i in range(n_classes):
    fpr2[i], tpr2[i], _ = roc_curve(
        y_test_bin[:, i], model2.predict_proba(X_test_selected)[:, i])
    roc_auc2[i] = auc(fpr2[i], tpr2[i])

# Plot ROC
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr1[i], tpr1[i], color=color, lw=2,
             label='ROC Modelo1 (AUC = %0.2f)' % roc_auc1[i])
    plt.plot(fpr2[i], tpr2[i], color=color, lw=2, linestyle='--',
             label='ROC Modelo2 (AUC = %0.2f)' % roc_auc2[i])
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC para Multiclase')
plt.legend(loc="lower right")
plt.show()
