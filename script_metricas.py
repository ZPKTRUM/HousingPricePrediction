import sys
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos del CSV
df = pd.read_csv('metricas_comparacion_modelos.csv')

# Eliminar columnas no deseadas
df = df.drop(columns=['Tiempo_Entrenamiento_s', 'Mejores_Parametros'])

# Redondear las columnas numéricas a 4 decimales
numeric_columns = ['CV_Accuracy_Mean', 'CV_Accuracy_Std', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']
df[numeric_columns] = df[numeric_columns].round(4)

# Combinar CV_Accuracy_Mean y CV_Accuracy_Std en una columna 'Accuracy'
df['Accuracy'] = df['CV_Accuracy_Mean'].astype(str) + ' ± ' + df['CV_Accuracy_Std'].astype(str)
df = df.drop(columns=['CV_Accuracy_Mean', 'CV_Accuracy_Std'])

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('tight')
ax.axis('off')

# Crear la tabla
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Ajustar el tamaño de la fuente y el espaciado
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.5, 2.0)  # Aumentar el ancho y alto de las celdas

# Ajustar el ancho de las columnas para que las palabras quepan
for (i, j), cell in table.get_celld().items():
    cell.set_text_props(wrap=True)
    if i == 0:  # Encabezados
        cell.set_height(0.1)
    else:  # Datos
        cell.set_height(0.08)

# Guardar la imagen como PNG
plt.savefig('metricas_tabla.png', bbox_inches='tight', dpi=300)
plt.close()