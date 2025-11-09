import sys
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos del CSV
df = pd.read_csv('Antiguo/Housing_price_prediction.csv')

# Crear diagramas de dispersión para variables clave
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Área vs Precio
axes[0,0].scatter(df['area'], df['price'])
axes[0,0].set_title('Área vs Precio')
axes[0,0].set_xlabel('Área (pies cuadrados)')
axes[0,0].set_ylabel('Precio')

# Dormitorios vs Precio
axes[0,1].scatter(df['bedrooms'], df['price'])
axes[0,1].set_title('Dormitorios vs Precio')
axes[0,1].set_xlabel('Número de Dormitorios')
axes[0,1].set_ylabel('Precio')

# Baños vs Precio
axes[1,0].scatter(df['bathrooms'], df['price'])
axes[1,0].set_title('Baños vs Precio')
axes[1,0].set_xlabel('Número de Baños')
axes[1,0].set_ylabel('Precio')

# Estacionamientos vs Precio
axes[1,1].scatter(df['parking'], df['price'])
axes[1,1].set_title('Estacionamientos vs Precio')
axes[1,1].set_xlabel('Número de Estacionamientos')
axes[1,1].set_ylabel('Precio')

plt.tight_layout()

# Guardar la imagen como PNG
plt.savefig('diagramas_dispersion.png', bbox_inches='tight', dpi=300)
plt.close()