import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import pandas as pd
from sklearn.cluster import KMeans           
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer 
import numpy as np  


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


def plot_silhouette_score(potential_k: list, silhouette_score_for_ks: list, model_name: str) -> None:
    """Plot the silhouette score with the value of k.

    Args:
        potential_k(list): k values to be used in the x-axis.
        silhouette_score_for_ks(list): silhouette score for each value in the potential_k.
        model_name(str): name of the used model that generated the silhouette score values.
    """ 
    plt.plot(potential_k, silhouette_score_for_ks)
    plt.xlabel('K') 
    plt.ylabel('Silhouette score')   
    plt.title('Silhouette Analysis For Optimal k Using ' + model_name)
    plt.show()
             

def plot_silhouette(n_clusters, model, data) -> None:
    """
    """

    #Check if the model is KMeans model to determine n_init in the instance 
    if model is KMeans:
        model_instance = model(n_clusters = n_clusters, n_init = "auto", random_state = 42)
    else:
        model_instance = model(n_clusters = n_clusters, random_state = 42)  

     #define a visualizer instance  
    visualizer = SilhouetteVisualizer(model_instance, colors='yellowbrick')

    # Fit the data to the visualizer
    visualizer.fit(data)

    # Finalize and render the figure 
    visualizer.show()

 

def plot_elbow(data, model, potential_k_list) -> None:
    """
    """

    #define a visualizer instance
    K_means_visualizer_reduced_data = KElbowVisualizer(model, k=potential_k_list, timings=False)

    # Fit the data to the visualizer
    K_means_visualizer_reduced_data.fit(data)

    # Finalize and render the figure 
    K_means_visualizer_reduced_data.show() 
  

def train_model(model, n_clusters: int, data: pd.DataFrame): 
    """Train a clustering model using the given data with the given number of clusters.

    Args:
        model: a clustering model.
        n_clusters(int): number of clusters to be used in the training process.
        data(pd.DataFrame): the dataframe taht contains data to be used in the training preocess.

    Returns:
        A label for each sample in the given dataset and centroids for each cluster.

    """
    #Check if the model is KMeans model to determine n_init in the instance 
    if model is KMeans:
        model_instance = model(n_clusters = n_clusters, n_init = "auto")
    else:
        model_instance = model(n_clusters = n_clusters)

    #predict labels
    labels = model_instance.fit_predict(data)
    
    #get clusters centroids
    centers = model_instance.cluster_centers_

    return labels, centers


def plot_2d_scatter(data: pd.DataFrame, x_axis_column_name: str, y_axis_column_name: str, labels: np.ndarray, embedded_centers: np.ndarray) -> None: 
    """Plot a 2d scatter plot of the given data each sample will be colored based on its cluster label.

    Args:
        data(pd.DataFrame): the dataframe taht contains data to be used in x-axis and y-axis.   
        embedded_centers(np.ndarray): An array of shape (n, 2) contains clusters centers, where n is the number of clusters.
        labels(np.array): an np.array contains label for each sample in the given dataset.
        y_axis_column_name(str):The name of the column, in the data dataframe, it's values will be used in the y-axis. 
        x_axis_column_name(str):The name of the column, in the data dataframe, it's values will be used in the x-axis. 
    """

    #create a new plot with the given figure size and style.
    plt.figure(figsize=(10, 10))
    plt.style.use("fivethirtyeight")

    # Plot a 2D scatter plot with the given x and y axis columns from the input data.
    scat = sns.scatterplot(
    x=x_axis_column_name,
    y=y_axis_column_name, 
    s=50,
    data=data, 
    hue=labels,  
    palette="Set2"
    )

    # Plot black dots at the values of each embedded center for clusters.
    plt.scatter(embedded_centers[:, 0], embedded_centers[:, 1], c='black', s=200, alpha=0.5)

    #change the plot title 
    scat.set_title( "Clustering results from .... data") 

    #display the plot
    plt.show()

 