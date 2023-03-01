def calculate_silhouette_score(model_instance, potential_k, data):
    silhouette_score_result = []
    # potential_k = list(range(min(potential_k_range), max(potential_k_range)))
    for k in potential_k:
        model = model_instance(n_clusters=k)
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
    
 