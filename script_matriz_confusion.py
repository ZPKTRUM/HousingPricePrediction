import sys
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos del CSV
df = pd.read_csv('Antiguo/mejor_modelo_matriz_confusion.csv')

# Extraer la matriz de confusión
cm = df.iloc[:, 1:].values

# Etiquetas
labels = ['Barata', 'Cara']

# Crear la figura
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusión - Mejor Modelo')
plt.colorbar()

# Etiquetas
tick_marks = range(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

# Anotar valores
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('Real')
plt.xlabel('Predicción')

# Guardar la imagen como PNG
plt.savefig('matriz_confusion.png', bbox_inches='tight', dpi=300)
plt.close()