from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

cluster_col = 'Cluster'

def plot_elbow_silhouette(data, max_k=20):
    def calculate_scores(k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        inertia = kmeans.inertia_  # Inercia de KMeans
        silhouette = silhouette_score(data, cluster_labels)  # Puntaje de Silhouette
        return inertia, silhouette

    scores = [calculate_scores(k) for k in range(2, max_k + 1)]
    inertia_scores, silhouette_scores = zip(*scores)

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), inertia_scores, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    
    plt.show()

def apply_clustering(data, method='kmeans', **kwargs):
    if method == 'kmeans':
        k = kwargs.get('n_clusters', 4)
        model = KMeans(n_clusters=k, random_state=42)
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        raise ValueError("El método debe ser 'kmeans' o 'dbscan'")
    
    clusters = model.fit_predict(data)
    return clusters

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)
    return rf, accuracy

def calculate_pca_loadings(pca, feature_names):
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    return loadings

def calculate_pca_contributions(loadings):
    contributions = loadings.apply(
        lambda x: np.square(x) / np.sum(np.square(x)), axis=0
    )
    return contributions

def draw_convex_hull(ax, x_data, y_data, **kwargs):
    points = np.c_[x_data, y_data]
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    ax.fill(hull_points[:, 0], hull_points[:, 1], **kwargs)

def plot_clusters(pca_components, clusters, method='kmeans', k=None):
    fig, ax = plt.subplots(figsize=(15, 6))
    
    df = pd.DataFrame({
        'PCA1': pca_components[:, 0],
        'PCA2': pca_components[:, 1],
        cluster_col: clusters
    })
    sns.scatterplot(x='PCA1', y='PCA2', hue=cluster_col, data=df, palette='Set1', ax=ax)
    
    unique_clusters = np.unique(clusters)
    
    for cluster in unique_clusters:
        if cluster == -1:  # DBSCAN noise
            continue
        cluster_data = df[df[cluster_col] == cluster]
        # draw_convex_hull(
        #     ax, cluster_data['PCA1'], cluster_data['PCA2'], alpha=0.2
        # )
        # Verificar si hay al menos 3 puntos en el cluster
        if len(cluster_data) >= 3:
            draw_convex_hull(
                ax, cluster_data['PCA1'], cluster_data['PCA2'], alpha=0.2
            )
    
    title = f'{method.upper()} Clustering'
    if k is not None:
        title += f' (k = {k})'
    
    ax.set_title(title)
    ax.set_xlabel('Dim1 (PCA)')
    ax.set_ylabel('Dim2 (PCA)')
    plt.show()

def predict_clusters(rf, new_products, preprocessor):
    new_products_preprocessed = preprocessor.transform(new_products)
    predicted_clusters = rf.predict(new_products_preprocessed)
    return predicted_clusters

def main(df_features, method='kmeans', cluster_col=None, **kwargs):
    # Separar características y etiquetas (si las etiquetas están presentes en df_features)
    if cluster_col and cluster_col in df_features.columns:
        X = df_features.drop(columns=cluster_col)
        y = df_features[cluster_col]
    else:
        X = df_features
        y = None
    
    # Identificar columnas numéricas y categóricas
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    transformers = []
    if len(num_cols) > 0:
        transformers.append(('num', RobustScaler(), num_cols))
    if len(cat_cols) > 0:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols))
    
    # Preprocesar datos solo si hay columnas relevantes
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers)
        data_preprocessed = preprocessor.fit_transform(X)
    else:
        data_preprocessed = X.values
    
    # Método de clustering
    if method == 'kmeans':
        plot_elbow_silhouette(data_preprocessed)
    
    clusters = apply_clustering(data_preprocessed, method=method, **kwargs)
    
    # Entrenamiento de Random Forest si hay etiquetas
    if y is not None:
        rf, accuracy = train_random_forest(data_preprocessed, clusters)
        print(f'Precisión del Random Forest en el conjunto de prueba: {accuracy:.2f}')
    else:
        rf = None
        accuracy = None
    
    # PCA para visualización
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data_preprocessed)
    
    # Obtener los nombres de las características
    feature_names = []
    if len(num_cols) > 0:
        feature_names += num_cols.tolist()
    if len(cat_cols) > 0:
        feature_names += preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()
    
    loadings = calculate_pca_loadings(pca, feature_names)
    contributions = calculate_pca_contributions(loadings)
    
    print("Cargas de PCA:")
    print(loadings)
    print("\nContribuciones de las características a cada componente PCA:")
    print(contributions)
    
    # Graficar los clusters
    k = kwargs.get('n_clusters') if method == 'kmeans' else None
    plot_clusters(pca_components, clusters, method=method, k=k)
    
    return rf, preprocessor, loadings, contributions

# EJEMPLO DE USO
# Ejemplo con K-means
# rf_model_kmeans, preprocessor_kmeans, pca_loadings_kmeans, pca_contributions_kmeans = main(
#     df_features, method='kmeans', n_clusters=4
# )

# Ejemplo con DBSCAN
# rf_model_dbscan, preprocessor_dbscan, pca_loadings_dbscan, pca_contributions_dbscan = main(
#     df_features, method='dbscan', eps=0.5, min_samples=5
# )

# Ejemplo de predicción con nuevos datos
# new_products = pd.read_csv('ruta_a_nuevos_productos.csv')
# predicted_clusters = predict_clusters(rf_model_kmeans, new_products, preprocessor_kmeans)
# new_products['predicted_cluster'] = predicted_clusters

# Imprimir los resultados del PCA
# print("K-means PCA Loadings:")
# print(pca_loadings_kmeans)

# print("DBSCAN PCA Contributions:")
# print(pca_contributions_dbscan)
