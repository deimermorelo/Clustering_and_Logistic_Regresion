# Solucion al Taller Jerarquia
# Diplomado Python Aplicado a la Ingenieria UPB
# Autor: Deimer David Morelo Ospino
# ID: 502217
# Email: deimer.morelo@upb.edu.co

# Generar valores simulados que pertenecen a cuatro grupos

#Importamos las librerias que utilizaremos

# Librerias a utilizar para tratamiento de datos
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs as mb

# Librerias a utilizar para graficos
import matplotlib.pyplot as plt
from matplotlib import style as st

# Librerias a utilizar para el preprocesado y el modelado de los datos
from sklearn.cluster import AgglomerativeClustering as ac
from scipy.cluster.hierarchy import dendrogram as dd
from sklearn.preprocessing import scale as sc
from sklearn.metrics import silhouette_score as ss

# Libreria a utilizar para la configuracion de warnings
import warnings as wn

# Funcion para extraer informacion de un modelo AgglomerativeClustering 
# y representar su dendograma a traves de la funcion dendogram del modulo
# scipy.cluster.hierarchy 
def plot_dendrogram(model, **kwargs):    
    
    # Creamos la variable 'counts', la cual contendrá un array de ceros
    # que se encontrara en concordancia con el tamaño de la posicion inicial (0)
    # del nodo hijo (children).
    counts = np.zeros(model.children_.shape[0])
    
    # Creamos la variable 'n_samples' con el numero total de muestras
    # que se tendrán, lo cual constituye el tamaño del objeto 'labels' 
    n_samples = len(model.labels_)
    
    # Diseñamos un ciclo for para cada una de las parejas de merge 
    # (elementos del nodo children). Dentro de dicho ciclo verificaremos si 
    # 'child_idx' es menor al numero de muestras (n_samples), si esto se cumple 
    # vamos guardando en un contador el numero de nodos hijos; en caso contrario,
    # el contador almacenará lo que se encuentra en la variable 'counts' en la 
    # posición que corresponda a la diferencia entre el elemento (child_idx)
    # y el numero de muestras (n_samples). 
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                # Nodo hoja (leaf node) 
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]
                
        # Asignamos lo guardado en la variable current_count a cada indice (i) 
        # del ciclo for.
        counts[i] = current_count
     
    # Matriz de enlace
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot del dendograma
    dd(linkage_matrix, **kwargs)

# Simulamos los datos
# ==============================================================================
X, y = mb(
        n_samples    = 200, 
        n_features   = 2, 
        centers      = 4, 
        cluster_std  = 0.60, 
        shuffle      = True, 
        random_state = 0
       )

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1], 
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker    = 'o',
        edgecolor = 'black', 
        label= f"Grupo {i}"
    )
ax.set_title('Datos simulados')
ax.legend();

# Realizamos el escalado de datos
# ==============================================================================
X_scaled = sc(X)

# Definimos nuestros modelos
# ==============================================================================
# Calculamos el vinculo entre los datos mediante el metodo 'complete' teniendo
# en cuenta la distancia euclidiana
modelo_hclust_complete = ac(
                            affinity = 'euclidean',
                            linkage  = 'complete',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_complete.fit(X=X_scaled)

# Calculamos el vinculo entre los datos mediante el metodo 'average' teniendo
# en cuenta la distancia euclidiana
modelo_hclust_average = ac(
                            affinity = 'euclidean',
                            linkage  = 'average',
                            distance_threshold = 0,
                            n_clusters         = None
                        )
modelo_hclust_average.fit(X=X_scaled)

# Calculamos el vinculo entre los datos mediante el metodo 'ward' teniendo
# en cuenta la distancia euclidiana
modelo_hclust_ward = ac(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            distance_threshold = 0,
                            n_clusters         = None
                     )
modelo_hclust_ward.fit(X=X_scaled)

# Generamos los dendrogramas
# ==============================================================================
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
plot_dendrogram(modelo_hclust_average, color_threshold=0, ax=axs[0])
axs[0].set_title("Distancia euclídea, Linkage average")
plot_dendrogram(modelo_hclust_complete, color_threshold=0, ax=axs[1])
axs[1].set_title("Distancia euclídea, Linkage complete")
plot_dendrogram(modelo_hclust_ward, color_threshold=0, ax=axs[2])
axs[2].set_title("Distancia euclídea, Linkage ward")
plt.tight_layout();

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
altura_corte = 6
plot_dendrogram(modelo_hclust_ward, color_threshold=altura_corte, ax=ax)
ax.set_title("Distancia euclídea, Linkage ward")
ax.axhline(y=altura_corte, c = 'black', linestyle='--', label='altura corte')
ax.legend();
 
# Creamos el método silhouette para identificar el número óptimo de clusters
# ==============================================================================
range_n_clusters = range(2, 15)
valores_medios_silhouette = []

for n_clusters in range_n_clusters:
    modelo = ac(
                    affinity   = 'euclidean',
                    linkage    = 'ward',
                    n_clusters = n_clusters
             )

    cluster_labels = modelo.fit_predict(X_scaled)
    silhouette_avg = ss(X_scaled, cluster_labels)
    valores_medios_silhouette.append(silhouette_avg)
    
fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.plot(range_n_clusters, valores_medios_silhouette, marker='o')
ax.set_title("Evolución de media de los índices silhouette")
ax.set_xlabel('Número clusters')
ax.set_ylabel('Media índices silhouette');

# Modelo 'hclust_ward'
# ==============================================================================
modelo_hclust_ward = ac(
                            affinity = 'euclidean',
                            linkage  = 'ward',
                            n_clusters = 4
                     )
modelo_hclust_ward.fit(X=X_scaled)




