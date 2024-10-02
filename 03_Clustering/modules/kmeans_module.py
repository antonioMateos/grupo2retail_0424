import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

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

def view_pca_variance(data):
    """
    Aplica PCA al DataFrame y grafica la varianza explicada acumulada y por componente.
    
    Parameters:
    data (DataFrame): DataFrame con las características (features) sin incluir la columna 'Cluster'.
    """
    fig_size = (15, 10)  # Ajustar tamaño para dos gráficos uno encima del otro

    if 'Cluster' in data.columns:
        data = data.drop(columns=['Cluster'])

    data = data.select_dtypes(include='number')  # Quedarse solo con columnas numéricas

    # Aplicar PCA
    pca = PCA(n_components=None)  # Mantener todos los componentes
    pca_result = pca.fit_transform(data)

    # Obtener la varianza explicada en porcentaje
    variance_ratio = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

    # Crear una figura con dos subplots uno encima del otro
    fig, ax = plt.subplots(2, 1, figsize=fig_size)  # 2 filas, 1 columna

    # Gráfico 1: Varianza explicada acumulada
    ax[0].plot(np.cumsum(variance_ratio), color='orange')
    ax[0].set_xlabel('Número de Componentes')
    ax[0].set_ylabel('Varianza Explicada Acumulada (%)')
    ax[0].set_title('Varianza Explicada Acumulada')

    # Añadir línea horizontal roja en el 90%
    ax[0].axhline(y=90, color='red', linestyle='--', linewidth=1.5)

    # Añadir líneas verticales discontinuas en gris en cada componente
    for i in range(1, len(variance_ratio) + 1):
        ax[0].axvline(x=i, color='gray', linestyle='--', alpha=0.7)
    ax[0].grid(True)

    # Gráfico 2: Varianza explicada por componente
    ax[1].bar(range(1, len(variance_ratio) + 1), variance_ratio, alpha=0.7, color='orange', align='center')
    ax[1].set_ylabel('Porcentaje de Varianza Explicada')
    ax[1].set_xlabel('Número de Componente Principal')
    ax[1].set_title('Varianza Explicada por Componentes')

    # Añadir líneas verticales discontinuas en gris en cada componente
    for i in range(1, len(variance_ratio) + 1):
        ax[1].axvline(x=i, color='gray', linestyle='--', alpha=0.7)
    ax[1].grid(True)

    # Ajustar los espacios entre los subplots para evitar que se solapen
    plt.tight_layout()

    # Mostrar los gráficos
    plt.show()

    # Mostrar la varianza explicada en porcentaje para cada componente
    print("Varianza explicada por cada componente (%):", variance_ratio)

    return variance_ratio


def calculate_accumulated_variance(variance_ratio):
    # Calcular la varianza explicada acumulada
    accumulated_variance = np.cumsum(variance_ratio)
    
    # Crear un DataFrame con una fila y tantas columnas como componentes
    df = pd.DataFrame([accumulated_variance], 
                      columns=[f'Componente {i+1}' for i in range(len(variance_ratio))])
    
    return df

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
    
    # Calcular las cargas (loadings)
    loadings = pca.components_.T
    feature_names = data.columns
    n_components = loadings.shape[1]  # Obtener el número de componentes

    # Crear un diccionario para almacenar las cargas de cada componente
    loading_dict = {'Feature': feature_names}

    # Añadir las cargas para cada componente al diccionario
    for i in range(n_components):
        loading_dict[f'Loading_PC{i+1}'] = loadings[:, i]

    # Crear el DataFrame de importancia de características
    loadings_df = pd.DataFrame(loading_dict).sort_values(by=f'Loading_PC1', ascending=False)

    return pca, pca_df, loadings_df

# Visualizacion T-SNE clusters
def tsne_visualization(data, n_components=2, perplexity=30):
    '''
    data === df_with_pca -> solo con columnas de componentes
    '''
    # 1. Preparar los datos
    # Obtenemos columnas componentes y cluster
    pc_cols = data.columns.tolist()
    pc_cols.remove(cluster_col)
    components = data[pc_cols]
    clusters = data[cluster_col]

    # 2. Estandarizar los datos
    scaler = RobustScaler()
    components_scaled = scaler.fit_transform(components)

    # 3. Aplicar t-SNE
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(components_scaled)

    # 4. Crear un DataFrame con los resultados de t-SNE y los clusters
    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE_1', 'TSNE_2'])
    df_tsne[cluster_col] = clusters

    # 5. Visualizar los resultados
    plt.figure(figsize=(15, 5))
    scatter = plt.scatter(df_tsne['TSNE_1'], df_tsne['TSNE_2'], 
                        c=df_tsne['Cluster'], cmap='viridis', alpha=0.6)

    # Añadir leyenda
    plt.title('Visualizacion clusters 2D t-SNE')
    plt.xlabel('t-SNE Componente 1')
    plt.ylabel('t-SNE Componente 2')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.show()

# DESCRIPCION DE CLUSTERS
# Añadir información a items originales
# Ver grafico de barras de items por cluster
# Crear etiquetas explicativas de los clusters en funcion de su descripcion
def pca_and_cluster(df, n_clusters=3, n_components=2):
    
    # # Perform PCA
    pca, pca_df,   = perform_pca(df, n_clusters, n_components)
    
    # Añadir los resultados al DataFrame original
    df_with_pca = df.copy()
    # df_with_pca[pca_columns] = pca_df[pca_columns]
    df_with_pca[pca_df.columns.tolist()] = pca_df[pca_df.columns.tolist()]
    df_with_pca['Cluster'] = pca_df['Cluster']
    
    return df_with_pca, loadings_df

def create_cluster_descriptions(df_with_pca, importance_df):
    cluster_descriptions = {}
    
    for cluster in df_with_pca['Cluster'].unique():
        cluster_data = df_with_pca[df_with_pca['Cluster'] == cluster]
        
        # Calcular las medias de las características originales para cada cluster
        cluster_means = cluster_data.mean()
        
        # Seleccionar las características más importantes basadas en las importancias de las feat en random forest
        important_features = importance_df['Feature'] # --> filtramos fuera según threshold
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
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()