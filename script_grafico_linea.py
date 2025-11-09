import sys
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos del CSV de métricas
df = pd.read_csv('Antiguo/metricas_comparacion_modelos.csv')

# Algoritmos y sus métricas
algoritmos = df['Algoritmo'].tolist()
accuracy = df['Test_Accuracy'].tolist()
precision = df['Precision'].tolist()
recall = df['Recall'].tolist()
f1_score = df['F1_Score'].tolist()

# Crear la figura
plt.figure(figsize=(12, 8))

# Graficar líneas para cada métrica
plt.plot(algoritmos, accuracy, marker='o', label='Accuracy', linewidth=2)
plt.plot(algoritmos, precision, marker='s', label='Precision', linewidth=2)
plt.plot(algoritmos, recall, marker='^', label='Recall', linewidth=2)
plt.plot(algoritmos, f1_score, marker='d', label='F1-Score', linewidth=2)

plt.title('Comparación de Métricas por Algoritmo')
plt.xlabel('Algoritmo')
plt.ylabel('Valor de la Métrica')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar la imagen como PNG
plt.savefig('grafico_linea.png', bbox_inches='tight', dpi=300)
plt.close()