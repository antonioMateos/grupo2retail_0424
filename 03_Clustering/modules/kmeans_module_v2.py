import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier # No need
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split # No need
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler # No need

CLUSTER_COL = 'cluster'

# INIT KMEANS
def init_kmeans(k):
    return KMeans(n_clusters=k, random_state=42, init = "k-means++", n_init="auto")

# Elbow & Silhouette
def plot_elbow(data_scaled, ran=20):
  distortions = []

  for k in range(1, ran):
      # kmeans = KMeans(n_clusters=k, random_state=42, init="random", n_init="auto")
      kmeans = init_kmeans(k)
      kmeans.fit(data_scaled)
      distortions.append(kmeans.inertia_)

  fig = px.line(
      x=range(1, 20),
      y=distortions,
      title="Elbow method - Inertia",
      labels={"x": "k", "y": "Inertia"},
  )
  fig.show()

def plot_silhouette(data_scaled, ran=20):
  from sklearn.metrics import silhouette_score

  silhouette_scores = []

  for k in range(2, ran):
      # kmeans = KMeans(n_clusters=k, random_state=42, init = "k-means++", n_init="auto")
      kmeans = init_kmeans(k)
      kmeans.fit(data_scaled)
      silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))

  fig = px.line(
      x=range(2, 20),
      y=silhouette_scores,
      title="Silhouette score",
      labels={"x": "k", "y": "Silhouette score"},
  )
  fig.show()
# END Elbow & Silhouette

# KMEANS & PLOT Bars
def apply_kmeans(data, k):
    kmeans = init_kmeans(k)
    return kmeans.fit(data) # Returns fit model 
    # return kmeans.fit_predict(data) # Returns fit_predit model ??? 

def add_cluster_labels(feature_matrix, labels):
    labels = model.labels_
    return feature_matrix.assign(**{cluster_col: labels})

def apply_kmeans_and_plot(optimal_k, data, feature_matrix, model):
    # kmeans = apply_kmeans(data, optimal_k) # Ahora recibimos el model
    feature_matrix_with_clusters = add_cluster_labels(feature_matrix, model.labels_)
    fig = plot_cluster_sizes(feature_matrix_with_clusters)
    plt.show()
    # return kmeans, feature_matrix_with_clusters # El modelo ya lo tenemos
    return feature_matrix_with_clusters

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