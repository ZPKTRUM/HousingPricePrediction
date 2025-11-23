# =============================================================================
# ANÁLISIS DE CLUSTERING - SEGMENTACIÓN DE VIVIENDAS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# 1. CARGA DEL CSV REAL
# =============================================================================

print("=" * 80)
print("ANÁLISIS DE CLUSTERING - SEGMENTACIÓN DE VIVIENDAS")
print("=" * 80)

print("\n1. CARGA Y EXPLORACIÓN DEL DATASET REAL")
print("-" * 50)

# Cargar el dataset real
try:
    df = pd.read_csv('Housing_price_prediction.csv')
    print("Dataset cargado exitosamente")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'Housing_price_prediction.csv'")
    print("Asegurate de que el archivo esté en el mismo directorio")
    exit()

print(f"Dimensiones del dataset: {df.shape}")
print(f"\nPrimeras 5 filas:")
print(df.head())

print(f"\nEstadísticas descriptivas:")
print(df.describe())

# =============================================================================
# 2. PREPROCESAMIENTO DE DATOS
# =============================================================================

print("\n\n2. PREPROCESAMIENTO DE DATOS")
print("-" * 40)

# Codificar variables categóricas
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea', 'furnishingstatus']

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Variables categóricas codificadas")

# Seleccionar todas las variables para clustering
features_for_clustering = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 
                         'mainroad', 'guestroom', 'basement', 'hotwaterheating',
                         'airconditioning', 'parking', 'prefarea', 'furnishingstatus']

X = df[features_for_clustering].copy()

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Datos escalados - Forma: {X_scaled.shape}")
print(f"Variables utilizadas: {features_for_clustering}")

# =============================================================================
# 3. MÉTODO DEL CODO AUTOMÁTICO
# =============================================================================

print("\n\n3. MÉTODO DEL CODO - ANÁLISIS AUTOMÁTICO")
print("-" * 50)

def calculate_elbow_point(distortions):
    """Calcula automáticamente el punto de codo usando el método del ángulo"""
    k_values = range(1, len(distortions) + 1)
    elbows = []
    
    for i in range(1, len(distortions) - 1):
        # Vectores entre puntos
        v1 = np.array([k_values[i-1] - k_values[i], distortions[i-1] - distortions[i]])
        v2 = np.array([k_values[i+1] - k_values[i], distortions[i+1] - distortions[i]])
        
        # Calcular ángulo (en grados)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        elbows.append(angle)
    
    # El codo está donde el ángulo es más agudo (más pequeño)
    elbow_idx = np.argmin(elbows) + 2  # +2 porque empezamos en k=1
    return elbow_idx

# Calcular inercias para diferentes valores de k
distortions = []
K_range = range(1, 11)

print("Calculando inercias para diferentes valores de k:")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    distortions.append(kmeans.inertia_)
    print(f"K={k}: Inercia = {kmeans.inertia_:.2f}")

# Calcular punto de codo automáticamente
elbow_k = calculate_elbow_point(distortions)
print(f"Punto de codo automático: K = {elbow_k}")

# =============================================================================
# 4. ANÁLISIS DE SILHOUETTE AUTOMÁTICO
# =============================================================================

print("\n\n4. ANÁLISIS DE SILHOUETTE - EVALUACIÓN AUTOMÁTICA")
print("-" * 50)

silhouette_scores = []
k_range = range(2, 11)

print("Calculando Silhouette Scores:")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    # Interpretación del score
    if silhouette_avg > 0.7:
        interpretation = "Estructura fuerte"
    elif silhouette_avg > 0.5:
        interpretation = "Estructura razonable"
    elif silhouette_avg > 0.25:
        interpretation = "Estructura débil"
    else:
        interpretation = "Sin estructura significativa"
    
    print(f"K={k}: Silhouette Score = {silhouette_avg:.4f} - {interpretation}")

# Encontrar mejor K automáticamente
best_silhouette_k = k_range[np.argmax(silhouette_scores)]
best_silhouette_score = max(silhouette_scores)

print(f"Mejor K por Silhouette: {best_silhouette_k} (score: {best_silhouette_score:.4f})")

# =============================================================================
# 5. MÉTODO GAP STATISTIC
# =============================================================================

print("\n\n5. MÉTODO GAP STATISTIC - ANÁLISIS AUTOMÁTICO")
print("-" * 50)

def calculate_gap_statistic(X, k_max=10, n_refs=5, random_state=42):
    """Calcula Gap Statistic para determinar el número óptimo de clusters"""
    np.random.seed(random_state)
    
    gaps = np.zeros(k_max)
    sks = np.zeros(k_max)
    
    print("Calculando Gap Statistic...")
    
    for k in range(1, k_max + 1):
        # Cluster datos reales
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia_real = kmeans.inertia_
        
        # Cluster datos de referencia
        ref_inertias = []
        for _ in range(n_refs):
            # Generar datos de referencia uniformes
            random_data = np.random.uniform(0, 1, size=X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_ref.fit(random_data)
            ref_inertias.append(kmeans_ref.inertia_)
        
        # Calcular gap
        gap = np.mean(np.log(ref_inertias)) - np.log(inertia_real)
        gaps[k-1] = gap
        
        # Calcular desviación estándar
        sk = np.sqrt(np.mean((np.log(ref_inertias) - np.mean(np.log(ref_inertias)))**2))
        sks[k-1] = sk * np.sqrt(1 + 1/n_refs)
        
        print(f"K={k}: Gap = {gap:.4f}")
    
    return gaps, sks

# Calcular Gap Statistic
gap_scores, gap_errors = calculate_gap_statistic(X_scaled, k_max=10)

# Encontrar mejor K por Gap Statistic
gap_k_range = range(1, 11)
best_gap_k = None

for k in range(len(gap_scores)-1):
    if gap_scores[k] >= gap_scores[k+1] - gap_errors[k+1]:
        best_gap_k = k + 1
        break

if best_gap_k is None:
    best_gap_k = gap_k_range[np.argmax(gap_scores)]

print(f"Mejor K por Gap Statistic: K = {best_gap_k}")

# =============================================================================
# 6. ALGORITMO AUTOMÁTICO DE DECISIÓN DE K ÓPTIMO
# =============================================================================

print("\n\n6. DECISIÓN AUTOMÁTICA DEL K ÓPTIMO")
print("-" * 50)

def determine_optimal_k(elbow_k, best_silhouette_k, best_gap_k, best_silhouette_score):
    """Algoritmo automático para determinar K óptimo considerando los 3 métodos"""
    
    print("Aplicando algoritmo de decisión automática...")
    
    # Contar votos de cada método
    votes = {}
    for k in [elbow_k, best_silhouette_k, best_gap_k]:
        votes[k] = votes.get(k, 0) + 1
    
    # Encontrar K con más votos
    max_votes = max(votes.values())
    candidates = [k for k, v in votes.items() if v == max_votes]
    
    if len(candidates) == 1:
        # Un claro ganador
        optimal_k = candidates[0]
        reason = f"Concordancia entre {max_votes} de 3 métodos"
    else:
        # Empate, priorizar silhouette score si es bueno
        if best_silhouette_score > 0.5:
            optimal_k = best_silhouette_k
            reason = f"Empate entre métodos, priorizando silhouette score ({best_silhouette_score:.3f})"
        else:
            # Priorizar el más pequeño para simplicidad
            optimal_k = min(candidates)
            reason = f"Empate entre métodos, seleccionando K más pequeño"
    
    return optimal_k, reason

# Aplicar algoritmo automático
optimal_k, selection_reason = determine_optimal_k(
    elbow_k, best_silhouette_k, best_gap_k, best_silhouette_score
)

print(f"RESULTADO AUTOMÁTICO:")
print(f"- Método del codo: K = {elbow_k}")
print(f"- Análisis de silhouette: K = {best_silhouette_k}")
print(f"- Gap Statistic: K = {best_gap_k}")
print(f"- Mejor silhouette score: {best_silhouette_score:.4f}")
print(f"- K óptimo seleccionado: {optimal_k}")
print(f"- Razón: {selection_reason}")

# =============================================================================
# 7. VISUALIZACIÓN COMPARATIVA DE LOS 3 MÉTODOS
# =============================================================================

print("\n\n7. VISUALIZACIÓN COMPARATIVA DE MÉTODOS")
print("-" * 50)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Método del codo
axes[0].plot(K_range, distortions, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Inercia')
axes[0].set_title('Método del Codo')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=elbow_k, color='red', linestyle='--', label=f'Codo: K={elbow_k}')
axes[0].legend()

# Análisis de Silhouette (GRÁFICO DE LÍNEA)
axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Análisis de Silhouette')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=best_silhouette_k, color='red', linestyle='--', 
                label=f'Mejor: K={best_silhouette_k}')
axes[1].legend()

# Gap Statistic
axes[2].plot(gap_k_range, gap_scores, 'go-', linewidth=2, markersize=8)
axes[2].fill_between(gap_k_range, gap_scores - gap_errors, gap_scores + gap_errors, 
                     alpha=0.2, color='green')
axes[2].set_xlabel('Número de Clusters (k)')
axes[2].set_ylabel('Gap Statistic')
axes[2].set_title('Método Gap Statistic')
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=best_gap_k, color='red', linestyle='--', 
                label=f'Mejor: K={best_gap_k}')
axes[2].legend()

# Resaltar K óptimo seleccionado en todos los gráficos
for ax in axes:
    ax.axvline(x=optimal_k, color='green', linestyle='-', linewidth=3, alpha=0.5,
               label=f'K óptimo: {optimal_k}')
    ax.legend()

plt.tight_layout()
plt.show()

# =============================================================================
# 8. APLICACIÓN DE K-MEANS CON K ÓPTIMO AUTOMÁTICO
# =============================================================================

print("\n\n8. APLICACIÓN DE K-MEANS CON K ÓPTIMO AUTOMÁTICO")
print("-" * 50)

print(f"Aplicando K-means con K={optimal_k}...")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Añadir clusters al DataFrame
df['cluster'] = cluster_labels

print(f"K-means aplicado exitosamente con K={optimal_k}")
print(f"Distribución de clusters:")
cluster_counts = df['cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} propiedades ({count/len(df)*100:.1f}%)")

# =============================================================================
# 9. ANÁLISIS DE LOS CLUSTERS
# =============================================================================

print("\n\n9. ANÁLISIS DE LOS CLUSTERS IDENTIFICADOS")
print("-" * 50)

# Centros de clusters en escala original
cluster_centers_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers_original, columns=features_for_clustering)
centers_df['cluster'] = range(optimal_k)

print("Centros de clusters (valores originales):")
print(centers_df.round(2))

# Análisis comparativo detallado
print(f"COMPARATIVA DETALLADA ENTRE {optimal_k} CLUSTERS:")
print("=" * 60)

# Variables clave para análisis (incluyendo 'stories')
key_vars = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    
    print(f"CLUSTER {cluster} ({len(cluster_data)} propiedades, {len(cluster_data)/len(df)*100:.1f}%):")
    print("-" * 40)
    
    for var in key_vars:
        stats = cluster_data[var].describe()
        print(f"{var.upper():>12}: Media={stats['mean']:.0f} | Mediana={stats['50%']:.0f} | Min={stats['min']:.0f} | Max={stats['max']:.0f}")

# =============================================================================
# 10. VISUALIZACIÓN DE CLUSTERS
# =============================================================================

print("\n\n10. VISUALIZACIÓN DE CLUSTERS")
print("-" * 40)

# Boxplots para las variables clave
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

for i, var in enumerate(key_vars):
    if i < len(axes):
        boxplot_data = []
        cluster_labels_list = []
        
        for cluster in range(optimal_k):
            boxplot_data.append(df[df['cluster'] == cluster][var].values)
            cluster_labels_list.append(f'Cluster {cluster}')
        
        bp = axes[i].boxplot(boxplot_data, 
                            labels=cluster_labels_list,
                            patch_artist=True,
                            showmeans=True,
                            meanline=True)
        
        # Colorear las cajas
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.7)
        
        axes[i].set_title(f'Distribución de {var.upper()}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(var, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Visualización con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                      cmap='viridis', alpha=0.7, s=50, edgecolors='w', linewidth=0.5)

# Centros de clusters en espacio PCA
centers_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200,
            label='Centros de Clusters', edgecolors='black', linewidth=2)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.title(f'Visualización de {optimal_k} Clusters con PCA')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Varianza explicada por PCA: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
      f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

# =============================================================================
# 11. ANÁLISIS DE SILHOUETTE PARA K ÓPTIMO
# =============================================================================

print("\n\n11. ANÁLISIS DE SILHOUETTE DETALLADO")
print("-" * 50)

# Calcular silhouette scores por muestra
sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
silhouette_avg = silhouette_score(X_scaled, cluster_labels)

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
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontweight='bold')
    y_lower = y_upper + 10

plt.title(f'Análisis de Silhouette para K={optimal_k}\n(Score promedio: {silhouette_avg:.3f})')
plt.xlabel('Coeficiente de Silhouette')
plt.ylabel('Cluster')
plt.axvline(x=silhouette_avg, color="red", linestyle="--")
plt.grid(True, alpha=0.3)
plt.show()

print(f"Silhouette Score promedio: {silhouette_avg:.4f}")

# =============================================================================
# 12. RESUMEN EJECUTIVO
# =============================================================================

print("\n" + "=" * 80)
print("RESUMEN EJECUTIVO")
print("=" * 80)

print(f"ANÁLISIS DE CLUSTERING COMPLETADO")
print(f"Dataset: {df.shape[0]} propiedades, {len(features_for_clustering)} variables")
print(f"K óptimo seleccionado automáticamente: {optimal_k} clusters")
print(f"Silhouette score final: {silhouette_avg:.4f}")
print(f"Método de selección: {selection_reason}")

print(f"Distribución final de clusters:")
for cluster, count in cluster_counts.items():
    print(f"  Cluster {cluster}: {count} propiedades ({count/len(df)*100:.1f}%)")

print(f"Variables analizadas: {', '.join(features_for_clustering)}")
print(f"Técnicas aplicadas: K-means, PCA, Método del Codo, Análisis de Silhouette, Gap Statistic")

print("\n" + "=" * 80)