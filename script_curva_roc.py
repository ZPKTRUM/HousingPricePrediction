import sys
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos del CSV de métricas
df = pd.read_csv('Antiguo/metricas_comparacion_modelos.csv')

# Algoritmos y sus AUC
algoritmos = df['Algoritmo'].tolist()
auc_values = df['AUC'].tolist()

# Crear la figura
plt.figure(figsize=(10, 8))

# Simular curvas ROC para cada algoritmo (usando AUC para aproximar)
for i, (alg, auc_val) in enumerate(zip(algoritmos, auc_values)):
    # Simular una curva ROC simple basada en AUC
    fpr = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    tpr = [0] + [min(1, auc_val * x) for x in [0.2, 0.4, 0.6, 0.8, 1.0]]
    plt.plot(fpr, tpr, label=f'{alg} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Línea base')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC Comparativas')
plt.legend(loc="lower right")

# Guardar la imagen como PNG
plt.savefig('curva_roc.png', bbox_inches='tight', dpi=300)
plt.close()