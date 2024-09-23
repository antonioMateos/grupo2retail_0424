import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull

cluster_col = 'Cluster'

def calculate_kmeans_scores(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(data, cluster_labels)
    return inertia, silhouette

def plot_elbow_silhouette(data, max_k=10):
    scores = [calculate_kmeans_scores(data, k) for k in range(2, max_k + 1)]
    inertia_scores, silhouette_scores = zip(*scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(range(2, max_k + 1), inertia_scores, 'bx-')
    ax1.set_xlabel('Número de Clusters (k)')
    ax1.set_ylabel('Inercia')
    ax1.set_title('Elbow Method')

    ax2.plot(range(2, max_k + 1), silhouette_scores, 'bo-')
    ax2.set_xlabel('Número de Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Method')
    
    plt.tight_layout()
    plt.show()

def apply_kmeans(data, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    return kmeans.fit(data)

def add_cluster_labels(feature_matrix, kmeans_labels):
    return feature_matrix.assign(**{cluster_col: kmeans_labels})

def plot_cluster_sizes(feature_matrix):
    cluster_counts = feature_matrix[cluster_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.tab10(range(len(cluster_counts))))
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', 
                ha='center', va='bottom')

    ax.set_xlabel(cluster_col)
    ax.set_ylabel('Número de items')
    ax.set_title('Número de items por cluster')
    ax.set_xticks(range(len(cluster_counts)))
    ax.set_xticklabels([f'Cluster {i}' for i in range(len(cluster_counts))], rotation=0)
    return fig

def apply_kmeans_and_plot(optimal_k, data, feature_matrix):
    kmeans = apply_kmeans(data, optimal_k)
    feature_matrix_with_clusters = add_cluster_labels(feature_matrix, kmeans.labels_)
    fig = plot_cluster_sizes(feature_matrix_with_clusters)
    plt.show()
    return kmeans, feature_matrix_with_clusters

def pca_visualization_2d(df):
    # Check if cluster_col exists in df
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in the DataFrame")

    # Separate features and cluster labels
    features = df.drop(columns=[cluster_col])
    cluster_labels = df[cluster_col]

    # Perform PCA
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(data=pca_features, columns=['PCA1', 'PCA2'])
    
    # Add cluster labels to the PCA results
    pca_df[cluster_col] = cluster_labels.values
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df[cluster_col], cmap='viridis')
    plt.colorbar(scatter)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_title('Visualización de Clusters en 2D con PCA')
    
    # Display the plot
    plt.show()
    
    return pca_df

# Helper function to perform PCA (if needed elsewhere)
def perform_pca(df, n_components=2):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(df)
    return pd.DataFrame(data=pca_features, columns=[f'PCA{i+1}' for i in range(n_components)])

def calculate_pca_variance(df, n_components=None):
    n_components = n_components or df.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(df)
    return pca.explained_variance_ratio_

def plot_pca_variance(explained_variance):
    fig, ax = plt.subplots(figsize=(12, 5))
    cumsum_variance = np.cumsum(explained_variance)
    ax.plot(cumsum_variance, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Número de Componentes')
    ax.set_ylabel('Varianza Explicada Acumulada')
    ax.set_title('Varianza Explicada por Componentes Principales')

    # Uso de escala logarítmica para el eje Y
    ax.set_yscale('log')

    for i, (var, cum_var) in enumerate(zip(explained_variance, cumsum_variance)):
        ax.text(i, cum_var + 0.02, f'{cum_var:.2f}', ha='center', va='bottom', fontsize=10)

    ax.grid(True)
    return fig

def pca_variance_plot(df, n_components=None):
    explained_variance = calculate_pca_variance(df, n_components)
    for i, var in enumerate(explained_variance):
        print(f"Principal Component {i+1}: {var:.2f}")
    fig = plot_pca_variance(explained_variance)
    # plt.show()
    return explained_variance

def calculate_feature_importances(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf.feature_importances_

def plot_feature_importances(importance_df, imp_threshold=0.05):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax.set_xlabel('Importancia de la característica')
    ax.set_ylabel('Características')
    ax.set_title('Importancia de las Características')
    ax.invert_yaxis()
    ax.axvline(x=imp_threshold, color='red', linestyle='--', linewidth=1, label=f'Importancia = {imp_threshold}')
    ax.legend()
    return fig

def get_feature_importances(df, clusters, imp_threshold=0.05):
    X = df.drop(columns=[cluster_col])
    y = df[cluster_col]
    importances = calculate_feature_importances(X, y)
    indices = np.argsort(importances)[::-1]
    importance_df = pd.DataFrame({
        'Feature': X.columns[indices],
        'Importance': importances[indices]
    })
    fig = plot_feature_importances(importance_df, imp_threshold)
    plt.show()
    return importance_df

def perform_pca(data, n_clusters, n_components=1):
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(data)
    
    # Create DataFrame with PCA results
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(data=pca_results, columns=pca_columns)
    
    # Apply KMeans clustering to the PCA results
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pca_df[cluster_col] = kmeans.fit_predict(pca_df)

    data_with_clusters = add_cluster_labels(data, kmeans.labels_)
    
    # Calculate loadings
    loadings = pca.components_.T
    feature_names = data.columns
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Loading_PC1': loadings[:, 0],
        'Loading_PC2': loadings[:, 1] if n_components > 1 else np.nan
    }).sort_values(by='Loading_PC1', ascending=False)
    
    if n_components < 2:
        importance_df = importance_df[['Feature', 'Loading_PC1']]

    return pca, pca_df, importance_df

def redux_dimensions_pca_and_cluster(df, n_clusters=3, n_components=2):
    # Separate features from the cluster column
    features = df.drop(columns=[cluster_col], errors='ignore')

    pca, pca_df, importance_df = perform_pca(features, n_clusters, n_components)

    # Create plots
    figs = plot_pca_results(pca_df, importance_df, n_components)
    plt.show()
    
    return pca_df, importance_df

def plot_pca_results(pca_df, importance_df, n_components):
    figs = []
    
    fig, ax = plt.subplots(figsize=(15, 5))
    importance_df[['Feature', 'Loading_PC1']].plot(kind='barh', x='Feature', y='Loading_PC1', color='skyblue', ax=ax, legend=False)
    ax.set_xlabel('Carga en PC1')
    ax.set_title('Importancia de las Características en el Primer Componente Principal')
    ax.invert_yaxis()
    ax.grid(True)
    figs.append(fig)

    if n_components > 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        importance_df[['Feature', 'Loading_PC2']].plot(kind='barh', x='Feature', y='Loading_PC2', color='lightgreen', ax=ax, legend=False)
        ax.set_xlabel('Carga en PC2')
        ax.set_title('Importancia de las Características en el Segundo Componente Principal')
        ax.invert_yaxis()
        ax.grid(True)
        figs.append(fig)

    fig, ax = plt.subplots(figsize=(15, 5))
    if n_components == 1:
        ax.scatter(pca_df['PC1'], np.zeros_like(pca_df['PC1']), c=pca_df[cluster_col], cmap='viridis', marker='o')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Valor (fijo en 0)')
        ax.set_title('Clustering después de PCA con 1 componente')
    elif n_components == 2:
        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df[cluster_col], cmap='viridis', marker='o')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('Clustering después de PCA con 2 componentes')
        plt.colorbar(scatter, label=cluster_col, ax=ax)
    
    ax.grid(True)
    figs.append(fig)
    
    return figs

# DESCRIPCION DE CLUSTERS
# Pendiente añadir información a items originales
# Ver grafico de barras de items por cluster
# Crear etiquetas explicativas de los clusters en funcion de su descripcion
def pca_and_cluster(df, n_clusters=3, n_components=2):
    
    # # Perform PCA
    pca, pca_df, importance_df = perform_pca(df, n_clusters, n_components)
    
    # Añadir los resultados al DataFrame original
    df_with_pca = df.copy()
    # df_with_pca[pca_columns] = pca_df[pca_columns]
    df_with_pca[pca_df.columns.tolist()] = pca_df[pca_df.columns.tolist()]
    df_with_pca['Cluster'] = pca_df['Cluster']
    
    return df_with_pca, importance_df

def create_cluster_descriptions(df_with_pca, importance_df):
    cluster_descriptions = {}
    
    for cluster in df_with_pca['Cluster'].unique():
        cluster_data = df_with_pca[df_with_pca['Cluster'] == cluster]
        
        # Calcular las medias de las características originales para cada cluster
        cluster_means = cluster_data.mean()
        
        # Seleccionar las características más importantes basadas en las importancias de las feat en los pca
        # important_features = importance_df.head(5)['Feature'] 
        important_features = importance_df['Feature'] # --> filtramos fuera según importancia
        important_means = cluster_means[important_features]
        
        description = pd.Series(important_means, name=f'Cluster {cluster}')
        cluster_descriptions[cluster] = description
    
    # Convertir el diccionario en un DataFrame
    description_df = pd.DataFrame(cluster_descriptions)
    
    return description_df

def plot_clusters_with_name(data):
    cluster_counts = data['cluster_name'].value_counts()

    # Crear una lista de colores, uno para cada barra utilizando la nueva sintaxis
    colors = plt.colormaps.get_cmap('tab10')  # Usar un colormap con diferentes colores

    # Crear el gráfico de barras
    plt.figure(figsize=(15,8))
    bars = plt.bar(cluster_counts.index, cluster_counts.values, color=[colors(i) for i in range(len(cluster_counts))])

    # Añadir etiquetas y título
    plt.title('Número de Elementos Únicos por Cluster Name', fontsize=16)
    plt.xlabel(' ', fontsize=14)

    # Añadir el número total de ítems en la parte superior de cada barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom', fontsize=14)

    # Ajustar rotación de etiquetas y márgenes
    plt.xticks(rotation=0, fontsize=14)
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()