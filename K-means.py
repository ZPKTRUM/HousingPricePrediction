# =============================================================================
# ANÁLISIS DE CLUSTERING - K-MEANS PARA SEGMENTACIÓN DE VIVIENDAS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

print("=" * 60)
print("ANÁLISIS DE CLUSTERING - SEGMENTACIÓN DE VIVIENDAS")
print("=" * 60)

# Cargar datos
df = pd.read_csv('Housing_price_prediction.csv')

# Seleccionar variables numéricas para clustering
numeric_vars = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
print(f"Variables numéricas seleccionadas: {numeric_vars}")

# Crear dataset para clustering
X_cluster = df[numeric_vars].copy()

print(f"\nDataset para clustering: {X_cluster.shape}")
print("\nEstadísticas descriptivas:")
print(X_cluster.describe())

# =============================================================================
# 2. ESCALADO DE DATOS
# =============================================================================

print("\n" + "=" * 60)
print("ESCALADO DE DATOS")
print("=" * 60)

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("Datos escalados - Estadísticas:")
print(f"Media: {X_scaled.mean(axis=0).round(2)}")
print(f"Desviación estándar: {X_scaled.std(axis=0).round(2)}")

# =============================================================================
# 3. MÉTODO DEL CODO (ELBOW METHOD)
# =============================================================================

print("\n" + "=" * 60)
print("MÉTODO DEL CODO - DETERMINANDO K ÓPTIMO")
print("=" * 60)

# Probar diferentes valores de k
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

    # Calcular silhouette score
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    print(
        f"K={k}: Inercia={inertias[-1]:.2f}, Silhouette={silhouette_avg:.4f}")

# Gráfico del método del codo
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.grid(True, alpha=0.3)

# Calcular la "rodilla" del codo
inertia_diffs = np.diff(inertias)
inertia_diff_ratios = inertia_diffs[1:] / inertia_diffs[:-1]
elbow_k = k_range[np.argmin(inertia_diff_ratios) + 2]
print(f"\nPosible codo en K = {elbow_k} (basado en ratios de diferencia)")

# =============================================================================
# 4. ANÁLISIS DE SILHOUETTE
# =============================================================================

plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Análisis de Silhouette')
plt.grid(True, alpha=0.3)

# Encontrar k con mejor silhouette score
best_silhouette_k = k_range[np.argmax(silhouette_scores)]
best_silhouette_score = max(silhouette_scores)
print(
    f"Mejor K por Silhouette: {best_silhouette_k} (score: {best_silhouette_score:.4f})")

# =============================================================================
# 5. MÉTODO GAP STATISTIC (SIMPLIFICADO)
# =============================================================================


def calculate_gap_statistic(X, k_max=10, n_refs=10):
    """Calcula Gap statistic simplificado"""
    gaps = []

    for k in range(1, k_max + 1):
        # Cluster datos reales
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia_real = np.log(kmeans.inertia_)

        # Cluster datos de referencia (uniformes)
        ref_inertias = []
        for _ in range(n_refs):
            # Generar datos de referencia uniformes
            random_data = np.random.uniform(0, 1, size=X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_ref.fit(random_data)
            ref_inertias.append(np.log(kmeans_ref.inertia_))

        gap = np.mean(ref_inertias) - inertia_real
        gaps.append(gap)

    return gaps


print("\nCalculando Gap Statistic...")
gap_scores = calculate_gap_statistic(X_scaled, k_max=10)
gap_k_range = range(1, 11)

plt.subplot(1, 3, 3)
plt.plot(gap_k_range, gap_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Gap Statistic')
plt.title('Método Gap Statistic')
plt.grid(True, alpha=0.3)

# Encontrar k con mayor gap
best_gap_k = gap_k_range[np.argmax(gap_scores)]
print(f"Mejor K por Gap Statistic: {best_gap_k}")

plt.tight_layout()
plt.savefig('metodos_seleccion_k.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. SELECCIÓN FINAL DE K Y APLICACIÓN DE K-MEANS
# =============================================================================

print("\n" + "=" * 60)
print("SELECCIÓN FINAL DE K Y APLICACIÓN DE K-MEANS")
print("=" * 60)

# Decidir k óptimo (priorizando silhouette score)
optimal_k = best_silhouette_k
print(f"K óptimo seleccionado: {optimal_k}")

# Aplicar K-means con k óptimo
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Añadir clusters al DataFrame original
df['cluster'] = cluster_labels
X_cluster['cluster'] = cluster_labels

print(f"\nDistribución de clusters:")
cluster_counts = df['cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} propiedades ({count/len(df)*100:.1f}%)")

# =============================================================================
# 7. ANÁLISIS DE CLUSTERS - BOXPLOTS TRADICIONALES MEJORADOS
# =============================================================================

print("\n" + "=" * 60)
print("ANÁLISIS DE CLUSTERS - BOXPLOTS")
print("=" * 60)

# Configuración para mejorar la visualización
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, var in enumerate(numeric_vars):
    # Crear boxplot básico pero con ajustes específicos
    boxplot_data = []
    cluster_labels_list = []
    
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster][var].values
        boxplot_data.append(cluster_data)
        cluster_labels_list.append(f'Cluster {cluster}')
    
    # Boxplot con parámetros ajustados para variables discretas
    bp = axes[i].boxplot(boxplot_data, 
                        labels=cluster_labels_list,
                        patch_artist=True,
                        showmeans=True,
                        meanline=True,
                        showfliers=True,
                        flierprops={'marker': 'o', 'markersize': 4, 'markerfacecolor': 'red'},
                        medianprops={'color': 'yellow', 'linewidth': 2},
                        meanprops={'marker': 'D', 'markerfacecolor': 'blue', 'markersize': 4})
    
    # Colorear las cajas
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[i].set_title(f'Distribución de {var.upper()} por Cluster', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Cluster', fontsize=12)
    axes[i].set_ylabel(var, fontsize=12)
    axes[i].grid(True, alpha=0.3, axis='y')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('boxplots_clusters_tradicionales.png', dpi=300, bbox_inches='tight')
plt.show()



# Informe de resumen de las distribuciones
print("\n" + "=" * 60)
print("RESUMEN ESTADÍSTICO POR CLUSTER")
print("=" * 60)

for var in numeric_vars:
    print(f"\n{var.upper()}:")
    stats = df.groupby('cluster')[var].agg(['min', 'max', 'mean', 'std']).round(2)
    print(stats)

# =============================================================================
# 8. CARACTERIZACIÓN DE CLUSTERS
# =============================================================================

print("\n" + "=" * 60)
print("CARACTERIZACIÓN DE CLUSTERS")
print("=" * 60)

# Calcular estadísticas por cluster
cluster_profiles = df.groupby('cluster')[numeric_vars].agg([
    'mean', 'std']).round(2)
print("\nPerfil de clusters (medias y desviaciones estándar):")
print(cluster_profiles)

# Centros de clusters en escala original
cluster_centers_original = scaler.inverse_transform(
    kmeans_final.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers_original, columns=numeric_vars)
centers_df['cluster'] = range(optimal_k)

print("\nCentros de clusters (valores originales):")
print(centers_df)

# =============================================================================
# 9. VISUALIZACIÓN EN 2D CON PCA
# =============================================================================

print("\n" + "=" * 60)
print("VISUALIZACIÓN CON PCA")
print("=" * 60)

# Aplicar PCA para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Crear DataFrame para visualización
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = cluster_labels

plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['cluster'],
                      cmap='viridis', alpha=0.7, s=50)

# Añadir centros de clusters en espacio PCA
centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200,
            label='Centros de Clusters')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.title('Visualización de Clusters con PCA')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('clusters_pca.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Varianza explicada por PCA: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# =============================================================================
# 10. ANÁLISIS DE SILHOUETTE PARA K ÓPTIMO
# =============================================================================

print("\n" + "=" * 60)
print("ANÁLISIS DETALLADO DE SILHOUETTE")
print("=" * 60)

# Calcular silhouette scores por muestra
sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

plt.figure(figsize=(10, 8))
y_lower = 10

for i in range(optimal_k):
    # Agrupar silhouette scores por cluster
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.viridis(float(i) / optimal_k)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Etiquetar clusters
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.title(f'Análisis de Silhouette para K={optimal_k}')
plt.xlabel('Coeficiente de Silhouette')
plt.ylabel('Cluster')
plt.axvline(x=silhouette_avg, color="red", linestyle="--",
            label=f'Score promedio: {silhouette_avg:.3f}')
plt.legend()
plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 11. INTERPRETACIÓN DE CLUSTERS
# =============================================================================

print("\n" + "=" * 60)
print("INTERPRETACIÓN DE CLUSTERS")
print("=" * 60)

# Crear perfiles interpretativos
cluster_descriptions = []

for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]

    # Características promedio
    avg_price = cluster_data['price'].mean()
    avg_area = cluster_data['area'].mean()
    avg_bedrooms = cluster_data['bedrooms'].mean()
    avg_bathrooms = cluster_data['bathrooms'].mean()

    # Determinar tipo de propiedad
    if avg_price > df['price'].quantile(0.75):
        price_segment = "Alto Valor"
    elif avg_price > df['price'].median():
        price_segment = "Valor Medio-Alto"
    elif avg_price > df['price'].quantile(0.25):
        price_segment = "Valor Medio-Bajo"
    else:
        price_segment = "Económico"

    if avg_area > df['area'].quantile(0.75):
        size_segment = "Grande"
    elif avg_area > df['area'].median():
        size_segment = "Mediana"
    else:
        size_segment = "Pequeña"

    description = f"Cluster {cluster}: {price_segment}, {size_segment}"
    cluster_descriptions.append(description)

    print(f"{description}")
    print(f"  - Precio promedio: ${avg_price:,.0f}")
    print(f"  - Área promedio: {avg_area:.0f} pies²")
    print(f"  - Dormitorios: {avg_bedrooms:.1f}")
    print(f"  - Baños: {avg_bathrooms:.1f}")
    print(f"  - Propiedades: {len(cluster_data)}")
    print()

# =============================================================================
# 12. GUARDAR RESULTADOS
# =============================================================================

print("\n" + "=" * 60)
print("GUARDANDO RESULTADOS")
print("=" * 60)

# Guardar datos con clusters
df_with_clusters = df.copy()
df_with_clusters.to_csv('viviendas_con_clusters.csv',
                        index=False, encoding='utf-8')
print("Archivo guardado: viviendas_con_clusters.csv")

# Guardar centros de clusters
centers_df.to_csv('centros_clusters.csv', index=False, encoding='utf-8')
print("Archivo guardado: centros_clusters.csv")

# Guardar métricas de selección de k
k_metrics = pd.DataFrame({
    'k': list(k_range),
    'inertia': inertias,
    'silhouette': silhouette_scores
})
k_metrics.to_csv('metricas_seleccion_k.csv', index=False, encoding='utf-8')
print("Archivo guardado: metricas_seleccion_k.csv")

# =============================================================================
# 13. RESUMEN EJECUTIVO
# =============================================================================

print("\n" + "=" * 60)
print("RESUMEN EJECUTIVO")
print("=" * 60)

print(f"ANÁLISIS DE CLUSTERING COMPLETADO")
print(
    f"Dataset: {X_cluster.shape[0]} propiedades, {X_cluster.shape[1]-1} variables")
print(f"K óptimo seleccionado: {optimal_k} clusters")
print(f"Silhouette score: {silhouette_avg:.4f}")
print(f"Método de selección: Combinación Elbow + Silhouette + Gap")
print(f"Clusters identificados: {', '.join(cluster_descriptions)}")

print(f"\nARCHIVOS GENERADOS:")
print("viviendas_con_clusters.csv")
print("centros_clusters.csv")
print("metricas_seleccion_k.csv")
print("metodos_seleccion_k.png")
print("boxplots_clusters.png")
print("clusters_pca.png")
print("silhouette_analysis.png")
