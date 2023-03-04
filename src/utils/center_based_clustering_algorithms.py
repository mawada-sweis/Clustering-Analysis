import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import pandas as pd
from sklearn.cluster import KMeans           
# from sklearn_extra.cluster import KMedoids


def calculate_silhouette_score(model, potential_k: list, data: pd.DataFrame) -> list:
    """Calculate the average silhouette score for each value in a potential_k list 
    for a given model using the given data.
    
    Args:
        data (pd.DataFrame):
        potential_k (list): k values we want to find the average score based on it.
        model (): 
 
    Return:
        (list): list of silhouette score results for each k in potential_k.
    """
    #define an empty list that will contain the avergae silhouette scores
    silhouette_score_result = [] 

    #calculate the avergae silhouette score for each k in the given list
    for k in potential_k:
        #Check if the model is KMeans model to determine n_init in the instance 
        if model is KMeans:
            model_instance = model(n_clusters = k, n_init = "auto", random_state = 42)
        else:
            model_instance = model(n_clusters = k, random_state = 42)

        #fit the model and get the predicted lable for each sample
        cluster_labels = model_instance.fit_predict(data)

        #calculate the silhouette score.
        silhouette_avg = silhouette_score(data, cluster_labels)

        #append the result of the average silhouette score 
        silhouette_score_result.append(silhouette_avg)

    return silhouette_score_result


def plot_silhouette_score(potential_k, silhouette_score_for_ks, model_name):
    """Plot a silhouete plot of the given model for the provided data

    Args:

    """
    plt.plot(potential_k, silhouette_score_for_ks)
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

 

def plot_elbow(data, model, potential_k_list):
    K_means_visualizer_reduced_data = KElbowVisualizer(model, k=potential_k_list, timings=False)
    K_means_visualizer_reduced_data.fit(data)# Fit the data to the visualizer
    K_means_visualizer_reduced_data.show() 
  



##### Selecting 5 means for k-medoids