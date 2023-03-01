import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


def calculate_silhouette_score(model_instance, potential_k, data):
    silhouette_score_result = [] 
    for k in potential_k:
        model = model_instance(n_init = "auto", n_clusters=k)
        cluster_labels = model.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_score_result.append(silhouette_avg)
    return silhouette_score_result



def plot_silhouette_score(potential_k_range, silhouette_score_for_ks, model_name):
    plt.plot(potential_k_range, silhouette_score_for_ks)
    plt.xlabel('K') 
    plt.ylabel('Silhouette score')   
    plt.title('Silhouette Analysis For Optimal k Using ' + model_name)
    plt.show()
    

def plot_silhoutte_k_mean(n_clusters, model, data):
    model_instance = model(n_clusters = n_clusters, n_init = "auto", random_state = 42)
    visualizer = SilhouetteVisualizer(model_instance, colors='yellowbrick')
    visualizer.fit(data)# Fit the data to the visualizer
    visualizer.show()# Finalize and render the figure 

def plot_silhoutte_k_medoids(n_clusters, model, data):
    model_instance = model(n_clusters = n_clusters, random_state = 42)
    visualizer = SilhouetteVisualizer(model_instance, colors='yellowbrick')
    visualizer.fit(data)# Fit the data to the visualizer
    visualizer.show()# Finalize and render the figure 

def plot_elbow(data, model, potential_k_list):
    K_means_visualizer_reduced_data = KElbowVisualizer(model, k=potential_k_list, timings=False)
    K_means_visualizer_reduced_data.fit(data)# Fit the data to the visualizer
    K_means_visualizer_reduced_data.show()# Finalize and render the figure
