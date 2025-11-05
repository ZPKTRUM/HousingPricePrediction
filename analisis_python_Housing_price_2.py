# =============================================================================
# ANALISIS COMPARATIVO DE 5 ALGORITMOS DE CLASIFICACION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================

print("=" * 60)
print("ANALISIS COMPARATIVO DE ALGORITMOS DE CLASIFICACION")
print("=" * 60)

# Cargar datos
df = pd.read_csv('Housing_price_prediction.csv')

# Crear variable objetivo binaria (clasificacion)
price_median = df['price'].median()
df['PriceCategory'] = df['price'].apply(lambda x: 'Cara' if x > price_median else 'Barata')

print(f"Dimensiones del dataset: {df.shape}")
print(f"Variable objetivo creada: PriceCategory")
print(f"   - Caras (>${price_median:,.0f}): {sum(df['PriceCategory'] == 'Cara')} propiedades")
print(f"   - Baratas (≤${price_median:,.0f}): {sum(df['PriceCategory'] == 'Barata')} propiedades")

# Codificar variables categoricas
le = LabelEncoder()
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea', 'furnishingstatus']

df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col])

df_encoded['PriceCategory'] = le.fit_transform(df_encoded['PriceCategory'])

# Preparar variables para modelado
X = df_encoded.drop(['price', 'PriceCategory'], axis=1)
y = df_encoded['PriceCategory']

print("\nVERIFICACION DE ESCALADO:")
print("Variables originales (primeras 5 filas):")
print(X.head())
print(f"\nRango de valores originales:")
for col in X.columns[:3]:  # Mostrar solo primeras 3 columnas
    print(f"   {col}: min={X[col].min():.2f}, max={X[col].max():.2f}")

# Estandarizar caracteristicas - IMPORTANTE PARA RED NEURONAL
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nVariables escaladas (primeras 5 filas):")
print(X_scaled[:5])
print(f"\nRango de valores escalados:")
print(f"   Min: {X_scaled.min():.2f}, Max: {X_scaled.max():.2f}")
print(f"   Media: {X_scaled.mean():.2f}, Desviacion: {X_scaled.std():.2f}")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, 
                                                   random_state=42, stratify=y)

print(f"\nConjunto de entrenamiento: {X_train.shape[0]} observaciones")
print(f"Conjunto de prueba: {X_test.shape[0]} observaciones")

# =============================================================================
# 2. DEFINICION DE MODELOS Y HIPERPARAMETROS
# =============================================================================

models = {
    'Regresion Logistica': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'Arbol de Decision': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Red Neuronal': {
        'model': MLPClassifier(random_state=42, max_iter=2000, early_stopping=True, validation_fraction=0.1),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'alpha': [0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01]
        }
    },
    'Naive Bayes': {
        'model': GaussianNB(),
        'params': {}
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    'K-Vecinos': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }
}

# =============================================================================
# 3. ENTRENAMIENTO Y VALIDACION CRUZADA
# =============================================================================

print("\n" + "=" * 60)
print("ENTRENAMIENTO Y OPTIMIZACION DE MODELOS")
print("=" * 60)

results = {}
training_times = {}
best_models = {}
tree_rules = {}
nn_loss_curves = {}

for name, config in models.items():
    print(f"Entrenando {name}...")
    
    start_time = time.time()
    
    if config['params']:  # Si tiene hiperparametros para optimizar
        grid_search = GridSearchCV(config['model'], config['params'], 
                                 cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        best_model = config['model']
        best_model.fit(X_train, y_train)
        best_params = "Sin parametros para optimizar"
    
    training_time = time.time() - start_time
    training_times[name] = training_time
    
    # Validacion cruzada
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Predicciones en test
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results[name] = {
        'model': best_model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'best_params': best_params,
        'training_time': training_time
    }
    
    best_models[name] = best_model
    
    # Guardar reglas del arbol de decision
    if name == 'Arbol de Decision':
        tree_rules['reglas_texto'] = export_text(best_model, feature_names=list(X.columns))
        tree_rules['caracteristicas_importancia'] = dict(zip(X.columns, best_model.feature_importances_))
    
    # Guardar curva de perdida de red neuronal
    if name == 'Red Neuronal' and hasattr(best_model, 'loss_curve_'):
        nn_loss_curves = best_model.loss_curve_
    
    print(f"   CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"   Tiempo entrenamiento: {training_time:.2f} segundos")
    print(f"   Mejores parametros: {best_params}")

# =============================================================================
# 4. ANALISIS DE RESULTADOS Y GRAFICOS
# =============================================================================

print("\n" + "=" * 60)
print("GENERANDO GRAFICOS DE RESULTADOS")
print("=" * 60)

# 4.1 Grafico comparativo de accuracy en validacion cruzada
plt.figure(figsize=(14, 10))

# Subplot 1: Accuracy en validacion cruzada
plt.subplot(2, 2, 1)
algorithms = list(results.keys())
cv_scores = [results[name]['cv_mean'] for name in algorithms]
cv_stds = [results[name]['cv_std'] for name in algorithms]

bars = plt.bar(algorithms, cv_scores, yerr=cv_stds, capsize=5, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title('Accuracy en Validacion Cruzada por Algoritmo', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)

# Anadir valores en las barras
for bar, score in zip(bars, cv_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 4.2 Grafico de tiempos de entrenamiento
plt.subplot(2, 2, 2)
times = [results[name]['training_time'] for name in algorithms]
bars = plt.bar(algorithms, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title('Tiempo de Entrenamiento por Algoritmo', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Tiempo (segundos)')

# Anadir valores en las barras
for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')

# 4.3 Grafico de F1-Score comparativo
plt.subplot(2, 2, 3)
f1_scores = [results[name]['f1'] for name in algorithms]
bars = plt.bar(algorithms, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title('F1-Score en Conjunto de Prueba', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('F1-Score')
plt.ylim(0.5, 1.0)

# Anadir valores en las barras
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 4.4 Grafico de AUC comparativo (solo para modelos que tienen AUC)
plt.subplot(2, 2, 4)
auc_scores = [results[name]['auc'] if results[name]['auc'] is not None else 0 for name in algorithms]
bars = plt.bar(algorithms, auc_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
plt.title('AUC en Conjunto de Prueba', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('AUC Score')
plt.ylim(0.5, 1.0)

# Anadir valores en las barras
for bar, score in zip(bars, auc_scores):
    if score > 0:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('comparacion_algoritmos.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.5 Curva de aprendizaje de la Red Neuronal
if nn_loss_curves is not None:
    plt.figure(figsize=(10, 6))
    plt.plot(nn_loss_curves, linewidth=2)
    plt.title('Curva de Aprendizaje - Red Neuronal', fontsize=16, fontweight='bold')
    plt.xlabel('Iteraciones')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.savefig('red_neuronal_curva_aprendizaje.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4.6 Curvas ROC comparativas
plt.figure(figsize=(12, 8))
for name, result in results.items():
    if result['auc'] is not None:
        model = result['model']
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = result['auc']
            plt.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Clasificador Aleatorio')
plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
plt.title('Curvas ROC - Comparacion de Algoritmos', fontsize=16, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.axis('equal')
plt.savefig('curvas_roc_comparativas.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.7 Matriz de confusion del mejor modelo
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Barata', 'Cara'], 
            yticklabels=['Barata', 'Cara'],
            annot_kws={"size": 14})
plt.title(f'Matriz de Confusion - {best_model_name}\n(Mejor Modelo)', 
         fontsize=16, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12)
plt.xlabel('Prediccion', fontsize=12)
plt.savefig('mejor_modelo_matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.8 Visualizacion del Arbol de Decision
best_tree_model = best_models['Arbol de Decision']
plt.figure(figsize=(20, 12))
plot_tree(best_tree_model, 
          feature_names=X.columns,
          class_names=['Barata', 'Cara'],
          filled=True,
          rounded=True,
          fontsize=10,
          impurity=False,
          proportion=True)
plt.title('Arbol de Decision - Visualizacion Completa', fontsize=18, fontweight='bold')
plt.savefig('arbol_decision_visualizacion.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.9 Importancia de variables del Arbol de Decision
plt.figure(figsize=(12, 8))
importancias = best_tree_model.feature_importances_
indices = np.argsort(importancias)[::-1]

plt.barh(range(len(importancias)), importancias[indices], color='steelblue')
plt.yticks(range(len(importancias)), [X.columns[i] for i in indices])
plt.xlabel('Importancia de la Variable')
plt.title('Importancia de Variables - Arbol de Decision', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('arbol_decision_importancia.png', dpi=300, bbox_inches='tight')
plt.show()

# 4.10 Coeficientes de Regresion Logistica
lr_model = best_models['Regresion Logistica']
if hasattr(lr_model, 'coef_'):
    plt.figure(figsize=(12, 8))
    coefficients = pd.DataFrame({
        'Variable': X.columns,
        'Coeficiente': lr_model.coef_[0]
    }).sort_values('Coeficiente', key=abs, ascending=True)
    
    plt.barh(coefficients['Variable'], coefficients['Coeficiente'], color='darkorange')
    plt.xlabel('Valor del Coeficiente')
    plt.title('Coeficientes de Regresion Logistica', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('regresion_logistica_coeficientes.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 5. GUARDAR RESULTADOS EN CSV
# =============================================================================

print("\n" + "=" * 60)
print("GUARDANDO RESULTADOS EN ARCHIVOS CSV")
print("=" * 60)

# 5.1 Guardar metricas de comparacion
metrics_comparison = pd.DataFrame({
    'Algoritmo': list(results.keys()),
    'CV_Accuracy_Mean': [results[name]['cv_mean'] for name in results],
    'CV_Accuracy_Std': [results[name]['cv_std'] for name in results],
    'Test_Accuracy': [results[name]['accuracy'] for name in results],
    'Precision': [results[name]['precision'] for name in results],
    'Recall': [results[name]['recall'] for name in results],
    'F1_Score': [results[name]['f1'] for name in results],
    'AUC': [results[name]['auc'] if results[name]['auc'] is not None else np.nan for name in results],
    'Tiempo_Entrenamiento_s': [results[name]['training_time'] for name in results],
    'Mejores_Parametros': [str(results[name]['best_params']) for name in results]
})

metrics_comparison.to_csv('metricas_comparacion_modelos.csv', index=False, encoding='utf-8')
print("Archivo guardado: metricas_comparacion_modelos.csv")

# 5.2 Guardar datos del arbol de decision - CORREGIDO
if tree_rules:
    # Importancia de caracteristicas del arbol
    importancia_df = pd.DataFrame({
        'Variable': list(tree_rules['caracteristicas_importancia'].keys()),
        'Importancia': list(tree_rules['caracteristicas_importancia'].values())
    }).sort_values('Importancia', ascending=False)
    
    importancia_df.to_csv('arbol_decision_importancia_variables.csv', index=False, encoding='utf-8')
    print("Archivo guardado: arbol_decision_importancia_variables.csv")
    
    # Reglas del arbol en texto
    with open('arbol_decision_reglas.txt', 'w', encoding='utf-8') as f:
        f.write("REGLAS DEL ARBOL DE DECISION\n")
        f.write("=" * 50 + "\n\n")
        f.write(tree_rules['reglas_texto'])
    print("Archivo guardado: arbol_decision_reglas.txt")
    
    # Predicciones del arbol de decision - CORREGIDO
    arbol_model = best_models['Arbol de Decision']
    y_pred_arbol = arbol_model.predict(X_test)
    y_prob_arbol = arbol_model.predict_proba(X_test)
    
    # Crear DataFrame con indices secuenciales
    predicciones_arbol = pd.DataFrame({
        'Indice_Test': range(len(X_test)),  # Indices en conjunto de test
        'Prediccion_Clase': y_pred_arbol,
        'Probabilidad_Clase_0': y_prob_arbol[:, 0],
        'Probabilidad_Clase_1': y_prob_arbol[:, 1],
        'Clase_Real': y_test
    })
    
    predicciones_arbol['Prediccion_Correcta'] = predicciones_arbol['Prediccion_Clase'] == predicciones_arbol['Clase_Real']
    predicciones_arbol.to_csv('arbol_decision_predicciones.csv', index=False, encoding='utf-8')
    print("Archivo guardado: arbol_decision_predicciones.csv")

# 5.3 Guardar coeficientes de regresion logistica
lr_model = best_models['Regresion Logistica']
if hasattr(lr_model, 'coef_'):
    coefficients = pd.DataFrame({
        'Variable': X.columns,
        'Coeficiente': lr_model.coef_[0],
        'Coeficiente_Absoluto': np.abs(lr_model.coef_[0])
    }).sort_values('Coeficiente_Absoluto', ascending=False)
    
    coefficients.to_csv('regresion_logistica_coeficientes.csv', index=False, encoding='utf-8')
    print("Archivo guardado: regresion_logistica_coeficientes.csv")

# 5.4 Guardar matriz de confusion del mejor modelo
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

matriz_confusion_df = pd.DataFrame(cm, 
                                 index=['Real_Barata', 'Real_Cara'],
                                 columns=['Pred_Barata', 'Pred_Cara'])
matriz_confusion_df.to_csv('mejor_modelo_matriz_confusion.csv', encoding='utf-8')
print("Archivo guardado: mejor_modelo_matriz_confusion.csv")

# =============================================================================
# 6. RESUMEN FINAL Y VERIFICACION RED NEURONAL
# =============================================================================

print("\n" + "=" * 60)
print("VERIFICACION RED NEURONAL - ESCALADO")
print("=" * 60)

# Verificar que los datos esten correctamente escalados
print("Verificacion de escalado para Red Neuronal:")
print(f"   Media de X_train: {X_train.mean():.6f}")
print(f"   Desviacion estandar de X_train: {X_train.std():.6f}")
print(f"   Rango de X_train: [{X_train.min():.2f}, {X_train.max():.2f}]")

if 'Red Neuronal' in best_models:
    nn_model = best_models['Red Neuronal']
    print(f"\nConfiguracion Red Neuronal:")
    print(f"   Arquitectura: {nn_model.hidden_layer_sizes}")
    print(f"   Alpha (regularizacion): {nn_model.alpha}")
    print(f"   Funcion de activacion: {nn_model.activation}")
    print(f"   Learning rate: {nn_model.learning_rate}")
    print(f"   Iteraciones realizadas: {nn_model.n_iter_}")
    
    if hasattr(nn_model, 'loss_curve_'):
        final_loss = nn_model.loss_curve_[-1] if nn_model.loss_curve_ else 'N/A'
        print(f"   Loss final: {final_loss}")

print("\n" + "=" * 60)
print("RESUMEN EJECUTIVO")
print("=" * 60)

best_overall = max(results, key=lambda x: results[x]['f1'])
best_result = results[best_overall]

print(f"MEJOR ALGORITMO: {best_overall}")
print(f"   F1-Score: {best_result['f1']:.4f}")
print(f"   Accuracy: {best_result['accuracy']:.4f}")
print(f"   AUC: {best_result['auc']:.4f}" if best_result['auc'] else "   AUC: N/A")
print(f"   Parametros: {best_result['best_params']}")

print(f"\nMETRICAS EN TEST DE TODOS LOS MODELOS:")
for name, result in results.items():
    print(f"   {name:<20}: F1 = {result['f1']:.4f}, Acc = {result['accuracy']:.4f}")

print(f"\nINFORMACION TECNICA:")
print(f"   Procesador: Intel Core i7 10700H")
print(f"   Tiempo total entrenamiento: {sum(training_times.values()):.2f} segundos")
print(f"   Algoritmo mas rapido: {min(training_times, key=training_times.get)}")
print(f"   Algoritmo mas lento: {max(training_times, key=training_times.get)}")

print(f"\nARCHIVOS GENERADOS:")
print("   metricas_comparacion_modelos.csv")
print("   arbol_decision_importancia_variables.csv")
print("   arbol_decision_reglas.txt")
print("   arbol_decision_predicciones.csv")
print("   regresion_logistica_coeficientes.csv")
print("   mejor_modelo_matriz_confusion.csv")
print("   comparacion_algoritmos.png")
print("   red_neuronal_curva_aprendizaje.png")
print("   arbol_decision_visualizacion.png")
print("   arbol_decision_importancia.png")
print("   curvas_roc_comparativas.png")
print("   mejor_modelo_matriz_confusion.png")
print("   regresion_logistica_coeficientes.png")