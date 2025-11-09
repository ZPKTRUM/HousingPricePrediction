import sys
import pandas as pd
import matplotlib.pyplot as plt

# Lista de variables independientes
variables_independientes = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
    'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
]

# Variable dependiente
variable_dependiente = 'price'

# Crear DataFrame con las variables
data = {
    'Variable Independiente': variables_independientes,
    'Variable Dependiente': [variable_dependiente] * len(variables_independientes)
}
df = pd.DataFrame(data)

# Crear la tabla con matplotlib y guardarla como imagen PNG
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
plt.savefig('variables_tabla.png')
plt.close()