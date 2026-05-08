from sklearn.cluster import DBSCAN
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.metrics import pairwise_distances

# Cargar el archivo CSV

BASE_DIR = Path(__file__).resolve().parent.parent
archivo_csv = BASE_DIR / "outputs" / "processed_survey_sample.csv"
df = pd.read_csv(archivo_csv, encoding='latin1', sep=',', low_memory=False)
# Selección de columnas
columnas_interes = [
    "swd",            # aceras delimitadas
    "ws",             # ancho de aceras
    "pedFlow",        # Flujo peatonal
    "pvQ",            # Calidad pavimento
    "pedInfr",        # Infraestructura peatonal
    "Obstr",          # Obstáculos en acera
    "Pres_pol_c",     # Presencia de elementos de seguridad (policías, cámaras)
    "Crime",          # Crimen
    "Crashes",        # Accidentes
    "Lighting",       # Iluminación
    "Cs",             # Velocidad vehicular
    "ptStops",        # Paradas de buses
    "Tcontrl",        # Control de tráfico
    "trFlow",         # Flujo vehicular
    "crTime",         # Tiempo de cruce
    "trDens",         # Suelo comercial
    "insDens",        # Suelo institucional/administrativo
    "ResDens",        # Suelo residencial
    "Aesthetics",     # Estética de edificios
    "Trees",          # Presencia de árboles
    "Block",          # Largo de cuadras
    "noise",          # Nivel de ruido
    "StagH2o",        # Agua estancada
    "Slope",          # Pendiente
    "AirPol",         # Contaminación
    "Cleanless",      # Limpieza en las calles
    "mpgaDens",       # Densidad de monumentos y cosas atractivas
]
data = df[columnas_interes]
data = data.fillna(0)

# Paso 0: Convertir el DataFrame a un array de NumPy
data_np = data.to_numpy()  # o data.values
data_1 = data_np.copy()    # Copia para las distancias originales

# Normalizar los datos
scaler = StandardScaler()
data_np = scaler.fit_transform(data_np)

# Reducción de dimensionalidad con UMAP
reducer = umap.UMAP(n_components=20, n_neighbors=15, min_dist=0.1, random_state=42)
data_np = reducer.fit_transform(data_np)

# Evaluación de la conservación de la estructura
from sklearn.metrics import pairwise_distances
distancias_orig = pairwise_distances(data_1)
distancias_reducidas = pairwise_distances(data_np)

from scipy.stats import spearmanr
coef, _ = spearmanr(distancias_orig.flatten(), distancias_reducidas.flatten())
print(f"Coeficiente de correlación de Spearman entre distancias originales y proyectadas: {coef:.4f}")


# Clustering jerárquico
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster, linkage

Z = linkage(data_np, method='average', metric='chebyshev')  # Puedes probar también con 'ward' si lo prefieres
#average + chebyshev da 0.395 de silueta para 4
# cosine dio 0,49 para 3 clusters

# Clustering jerárquico - versión corregida
from scipy.cluster.hierarchy import dendrogram

# 2. Visualizar el dendrograma para elegir el número de clusters
plt.figure(figsize=(12, 6))
dendrogram(Z, p=10, truncate_mode='level', show_leaf_counts=True)
plt.title('Dendrograma del clustering jerárquico')
plt.xlabel('Índice del punto')
plt.ylabel('Distancia')
plt.axhline(y=Z[-4, 2], color='r', linestyle='--')  # Línea para 4 clusters
plt.show()




from scipy.cluster.hierarchy import fcluster

def buscar_umbral_aproximado(Z, n_deseados, max_iter=1000):
    """
    Intenta encontrar el umbral de distancia que produzca aproximadamente n_deseados clusters.
    Si no lo encuentra exactamente, entrega el más cercano.
    """
    dists = Z[:, 2]
    t_min, t_max = dists.min(), dists.max()
    mejor_t = None
    mejor_diff = float('inf')
    mejor_clusters = None

    for t in np.linspace(t_min, t_max, max_iter):
        clusters_temp = fcluster(Z, t=t, criterion='distance')
        n_clusters = len(np.unique(clusters_temp))
        diff = abs(n_clusters - n_deseados)

        if diff < mejor_diff:
            mejor_diff = diff
            mejor_t = t
            mejor_clusters = clusters_temp

        if n_clusters == n_deseados:
            return t, clusters_temp  # Lo encontramos exacto

    print(f"No se encontró exactamente {n_deseados} clusters. Lo más cercano fue {len(np.unique(mejor_clusters))} clusters.")
    return mejor_t, mejor_clusters

umbral_4, clusters = buscar_umbral_aproximado(Z, 4)


plt.figure(figsize=(8, 6))
plt.scatter(data_np[:, 0], data_np[:, 1], c=clusters, cmap='tab10', s=10)
plt.title('Clustering jerárquico con 4 clusters')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.grid(True)
plt.savefig(BASE_DIR / "images" / "umap_projection.png",
            dpi=300,
            bbox_inches='tight')
plt.show()




# Seleccionamos 4 clusters

k = 4
# Aplicar fcluster
clusters = fcluster(Z, t=k, criterion='maxclust')

# Verificar el número de clusters obtenidos
unique_clusters = np.unique(clusters)
print(f"Número de clusters obtenidos: {len(unique_clusters)}")
print(f"Distribución de puntos por cluster: {np.bincount(clusters)}")

from scipy.cluster.hierarchy import fcluster

# Exploramos las alturas donde se forman los clusters
for i in range(2, 10):
    clusters_temp = fcluster(Z, t=i, criterion='maxclust')
    print(f"Con t={i}, número de clusters: {len(np.unique(clusters_temp))}")



# Calcular el índice de silueta
if len(np.unique(clusters)) > 1:
    score = silhouette_score(data_np, clusters)
    print(f"Coeficiente de silueta: {score}")
else:
    print("No se pudo calcular el coeficiente de silueta (solo hay un cluster).")

# Opcional: evaluar diferentes números de clusters
cluster_range = range(2, 14)
silhouette_scores = []
for k in cluster_range:
    clusters_temp = fcluster(Z, t=k, criterion='maxclust')
    if len(np.unique(clusters_temp)) > 1:
        score_temp = silhouette_score(data_np, clusters_temp)
        silhouette_scores.append(score_temp)
    else:
        silhouette_scores.append(np.nan)
        
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Número de clusters')
plt.ylabel('Coeficiente de silueta')
plt.title('Coeficiente de silueta vs. Número de clusters')
plt.grid()
plt.show()



from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score 
dbi = davies_bouldin_score(data_np, clusters)
ch_score = calinski_harabasz_score(data_np, clusters)
print(f"Índice Davies-Bouldin: {dbi:.4f}")
print(f"Índice de Calinski-Harabasz: {ch_score:.4f}")

# Rango de número de clusters a evaluar
cluster_range = range(2, 14)

# Lista para almacenar los valores de Calinski–Harabasz
ch_scores = []

for k in cluster_range:
    # Generar clusters con 'maxclust'
    clusters_temp = fcluster(Z, t=k, criterion='maxclust')
    # Solo calcular si hay más de 1 cluster
    if len(np.unique(clusters_temp)) > 1:
        score_ch = calinski_harabasz_score(data_np, clusters_temp)
        ch_scores.append(score_ch)
    else:
        ch_scores.append(np.nan)

# Graficar Calinski–Harabasz vs. Número de clusters
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, ch_scores, marker='o', linestyle='-')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Índice de Calinski–Harabasz')
plt.title('Calinski–Harabasz vs. Número de clusters')
plt.grid(True)
plt.show()

# Agregar la columna de clusters al DataFrame original
data['Cluster_Jerarquico'] = clusters
df['Cluster_Jerarquico'] = clusters

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# Calcula matrices de distancia
dist_euc = squareform(pdist(data_np, metric='euclidean'))
dist_cos = squareform(pdist(data_np, metric='cosine'))

# Correlación entre ambas
#corr, _ = spearmanr(dist_euc.flatten(), dist_cos.flatten())
#print(f"Correlación Spearman entre distancias euclidianas y coseno: {corr:.4f}")

# Guardar el DataFrame en un nuevo archivo CSV
output_csv = BASE_DIR / "outputs" / "hierarchical_clusters_processed.csv"
output_csv2 = BASE_DIR / "outputs" / "hierarchical_clusters_full.csv"
data.to_csv(output_csv, index=False, sep=';', encoding='latin1', float_format="%.4f")
df.to_csv(output_csv2, index=False, sep=';', encoding='latin1', float_format="%.4f")
