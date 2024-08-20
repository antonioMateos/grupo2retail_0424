import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, make_scorer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer


# Lo hacemos fuera
def preprocess_data(feature_matrix):

    # Identificar columnas categóricas y numéricas
    categorical_columns = feature_matrix.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = feature_matrix.select_dtypes(include=[float, int]).columns.tolist()

    print(f'Cat cols: {len(categorical_columns)}')
    print(f'Num cols: {len(numeric_columns)}')

    # Definir la lista de transformadores
    transformers = []
    
    # Agregar el RobustScaler para las columnas numéricas si existen
    if numeric_columns:
        transformers.append(('num', RobustScaler(), numeric_columns))
    
    # Agregar el OneHotEncoder para las columnas categóricas si existen
    if categorical_columns:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse=False), categorical_columns))

    # Crear el preprocesador con los transformadores existentes
    preprocessor = ColumnTransformer(transformers)
    
    return preprocessor, preprocessor.fit_transform(feature_matrix)


def apply_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    silhouette_avg = silhouette_score(data, clusters)
    return clusters, silhouette_avg


def silhouette_per_cluster(data, clusters):
    sample_silhouette_values = silhouette_samples(data, clusters)
    df_silhouette = pd.DataFrame({
        'Cluster': clusters,
        'Silhouette Value': sample_silhouette_values
    })
    return df_silhouette.groupby('Cluster').mean()


def noise_ratio(clusters):
    noise_points = np.sum(clusters == -1)
    total_points = len(clusters)
    return noise_points / total_points


def stability_analysis(data, clusters_list):
    for i in range(len(clusters_list) - 1):
        for j in range(i + 1, len(clusters_list)):
            ari = adjusted_rand_score(clusters_list[i], clusters_list[j])
            nmi = normalized_mutual_info_score(clusters_list[i], clusters_list[j])
            print(f"ARI entre configuración {i+1} y {j+1}: {ari}")
            print(f"NMI entre configuración {i+1} y {j+1}: {nmi}")


def external_validation(clusters, true_labels):
    ari = adjusted_rand_score(true_labels, clusters)
    nmi = normalized_mutual_info_score(true_labels, clusters)
    return ari, nmi


def interpret_clusters(feature_matrix, clusters):
    feature_matrix['Cluster'] = clusters
    summary = feature_matrix.groupby('Cluster').agg(['mean', 'median'])
    return summary

def custom_scorer(estimator, X):
    """
    Calcula el Silhouette Score, ignorando los clusters de ruido (-1).
    """
    clusters = estimator.fit_predict(X)
    if len(set(clusters)) == 1 or all(cl == -1 for cl in clusters):
        # Evitar el caso donde hay un solo clúster o todo es ruido
        return -1
    return silhouette_score(X, clusters)


def random_search_dbscan(data, n_iter=50):

    param_distributions = {
        'eps': np.arange(0.1, 1.0, 0.01),
        'min_samples': range(2, 20)
    }

    dbscan = DBSCAN()

    random_search = RandomizedSearchCV(
        dbscan, param_distributions, n_iter=n_iter,
        scoring=make_scorer(custom_scorer), cv=3, random_state=42
    )
    
    random_search.fit(data)

    return random_search.best_params_, random_search.best_score_

def plot_pca_clusters(pca_components, clusters):
    fig, ax = plt.subplots(figsize=(7, 6))
    
    df = pd.DataFrame({
        'PCA1': pca_components[:, 0],
        'PCA2': pca_components[:, 1],
        'Cluster': clusters
    })
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Cluster', data=df,
        palette='Set1', ax=ax
    )
    
    unique_clusters = np.unique(clusters)
    
    for cluster in unique_clusters:
        if cluster == -1:  # DBSCAN noise
            continue
        cluster_data = df[df['Cluster'] == cluster]
        draw_convex_hull(
            ax, cluster_data['PCA1'], cluster_data['PCA2'], alpha=0.2
        )
    
    ax.set_title('DBSCAN Clustering')
    ax.set_xlabel('Dim1 (PCA)')
    ax.set_ylabel('Dim2 (PCA)')
    plt.show()


def draw_convex_hull(ax, x_data, y_data, **kwargs):
    from scipy.spatial import ConvexHull
    points = np.c_[x_data, y_data]
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    ax.fill(hull_points[:, 0], hull_points[:, 1], **kwargs)


def analyze_noise_points(feature_matrix, clusters):
    noise_points = feature_matrix[clusters == -1]
    return noise_points.describe()


def main(feature_matrix_renamed, eps=0.5, min_samples=5, true_labels=None):
    # Preprocesar los datos
    preprocessor, data_preprocessed = preprocess_data(
        feature_matrix_renamed
    )

    # Aplicar DBSCAN
    clusters, silhouette_avg = apply_dbscan(
        data_preprocessed, eps=eps, min_samples=min_samples
    )
    print(f'Silhouette Score: {silhouette_avg:.2f}')

    # Análisis del Silhouette por clúster
    silhouette_cluster = silhouette_per_cluster(data_preprocessed, clusters)
    print("\nSilhouette por Clúster:")
    print(silhouette_cluster)

    # Evaluación del ruido
    noise_ratio_value = noise_ratio(clusters)
    print(f"\nProporción de puntos clasificados como ruido: {noise_ratio_value:.2f}")

    # Evaluación de estabilidad (usando configuraciones simuladas)
    stability_analysis(data_preprocessed, [clusters])

    # Evaluación externa si hay etiquetas disponibles
    if true_labels is not None:
        ari, nmi = external_validation(clusters, true_labels)
        print(f"\nARI: {ari:.2f}, NMI: {nmi:.2f}")

    # Análisis de clusters
    cluster_summary = interpret_clusters(feature_matrix_renamed, clusters)
    print("\nResumen de los clusters:")
    print(cluster_summary)

    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data_preprocessed)

    # Visualización de los clusters
    plot_pca_clusters(pca_components, clusters)

    return feature_matrix_renamed, cluster_summary, clusters

# Ejemplo de uso
# Supongamos que ya tienes un DataFrame `feature_matrix_renamed`
# cargado con las características ya preprocesadas.

# feature_matrix_renamed = pd.read_csv('ruta_a_tu_feature_matrix.csv')

# Si tienes etiquetas verdaderas, puedes usarlas para validación externa:
# true_labels = pd.read_csv('ruta_a_las_etiquetas_verdaderas.csv')

# # Ejecutar el clustering con DBSCAN
# feature_matrix_renamed, cluster_summary, clusters = dbc.main(
#     feature_matrix_renamed, eps=0.6, min_samples=10, true_labels=None
# )

# # Análisis de los puntos de ruido
# noise_analysis = dbc.analyze_noise_points(feature_matrix_renamed, clusters)
# print("\nAnálisis de los puntos de ruido:")
# print(noise_analysis)

# # Ejemplo de búsqueda de hiperparámetros de DBSCAN
# best_params, best_score = dbc.grid_search_dbscan(feature_matrix_renamed)
# print(f"\nMejores parámetros de DBSCAN: {best_params}")
# print(f"Mejor Silhouette Score obtenido: {best_score:.2f}")