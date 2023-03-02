import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import pandas as pd
from sklearn.cluster import KMeans           
# from sklearn_extra.cluster import KMedoids


def calculate_silhouette_score(model, potential_k, data) -> list:
    """Word
    
    Args:
        model_dkje (pd.DataFrame): diojfijdoi
        oadjfosjdf;l
    
    Return:
        (int): sjdfivjhsfdi
    """
    silhouette_score_result = [] 
    for k in potential_k:
        if model is KMeans:
            model_instance = model(n_clusters = k, n_init = "auto", random_state = 42)
        else:
            model_instance = model(n_clusters = k, random_state = 42) 
        cluster_labels = model_instance.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_score_result.append(silhouette_avg)
    return silhouette_score_result


def plot_silhouette_score(potential_k_range, silhouette_score_for_ks, model_name):
    plt.plot(potential_k_range, silhouette_score_for_ks)
    plt.xlabel('K') 
    plt.ylabel('Silhouette score')   
    plt.title('Silhouette Analysis For Optimal k Using ' + model_name)
    plt.show()
             

def plot_silhouette(n_clusters, model, data):
    if model is KMeans:
        model_instance = model(n_clusters = n_clusters, n_init = "auto", random_state = 42)
    else:
        model_instance = model(n_clusters = n_clusters, random_state = 42)    
    visualizer = SilhouetteVisualizer(model_instance, colors='yellowbrick')
    visualizer.fit(data)# Fit the data to the visualizer
    visualizer.show()# Finalize and render the figure 

# def plot_silhoutte_k_medoids(n_clusters, model, data):
#     visualizer = SilhouetteVisualizer(model_instance, colors='yellowbrick')
#     visualizer.fit(data)# Fit the data to the visualizer
#     visualizer.show()# Finalize and render the figure 
 

def plot_elbow(data, model, potential_k_list):
    K_means_visualizer_reduced_data = KElbowVisualizer(model, k=potential_k_list, timings=False)
    K_means_visualizer_reduced_data.fit(data)# Fit the data to the visualizer
    K_means_visualizer_reduced_data.show() 




##### Selecting 5 means for k-medoids