# The Citywide Mobility Clustering

# Data Description
The Citywide Mobility Survey (CMS) is a survey conducted by the New York City Department of Transportation (DOT) to gather information about the travel behavior, preferences, and attitudes of New York City residents. The survey is conducted periodically, and the data collected is used to inform transportation planning and policy decisions.

# Table of Contents
* [Data Description](#data-description)

* [Data Objectives](#data-objectives) 

* [Project Objectives](#project-objective)

* [Hierarchical Clustering](#hierarchical-clustering)

* [K-Means Clustering](#k-Means-clustering)

* [DBSCAN Model](#DBSCAN-model)

# Data Objectives 
The primary objectives of the CMS data collection are:
* Identify the factors and experiences that drive transportation choices for New York City residents.
* Understand current views on the state of transportation within the City.
* Measure attitudes toward current transportation issues and topics in New York City.


# Project Objective
The objective of this project is to analyze the CMS dataset to gain insights into the travel behavior, preferences, and attitudes of New York City residents. Specifically, we aim to:

* Understand the relationship between residents' behaviors, preferences, attitudes, and traveling methods.
* Identify any trends or patterns in the data that may be relevant to transportation planning and policy decisions.


By achieving these objectives, we hope to contribute to the ongoing efforts to improve transportation in New York City and enhance the mobility of its residents.


# Project Process
We maintained just the data of the individuals who completed the survey, leaving out the remainder of the participating family members.
As indicated in Table (1), the data comprises two categories of missing data: natural missing data (NaN) and missing data due to particular reasons, with each category having a label that we processed differently. We maintained all columns except those that were eliminated following discussion:
Columns with more than 75% missing data
Columns with 65% missing data for a specified cause.


<div align="center">

| Label | Description
|------|-----|
| Not required under the circumstances | 995 
| Don't know | 998 
| Prefer not to answer | 999 
| No response for required field | -9998 
| A technological error occurred | -9999 

Table (1): Missing Data Labels
</div>

The data consists of naturally encoded columns, label data, numeric data, category data, and dates, all of which we encoded.
We scaled the data using the standard and minmax scalers. then for the final clustering, we chose the minmax scaler result since the models deal with it more efficiently than the standard scaler result. Clustering was done using three models: k-mean, k-medoid, DBSCAN, and hierarchical. After selecting the best model result, we examined each cluster and its important characteristics.

# Hierarchical Clustering
We examine the Ward linkage method's use in hierarchical clustering. The dendrogram, shown in Graph (1), determines the number of clusters to test in our analysis.

<div align="center">

![Graph(1): Dendrogram Presentation using Ward Linkage](https://user-images.githubusercontent.com/67907899/223112983-b358351f-33c5-44e9-b692-f5d8e2392dc5.png)

Graph(1): Dendrogram Presentation using Ward Linkage.
</div>

To determine the appropriate number of clusters to test, we examine the dendrogram and look for the longest distance where there is a clear separation of data points. Based on this analysis, we chose to test clustering scenarios with three and four clusters. This decision is reflected in Graph (2), which shows the dendrogram and the resulting clusters for our analysis. By selecting three and four clusters, we can compare the effectiveness of different clustering scenarios and identify the optimal number of clusters for our data.

<div align="center">

| Threshold at 60 then we have 3 clusters | Threshold at 55.5 then we have 4 clusters
|------|-----|
| ![Dendrogram at 60](https://user-images.githubusercontent.com/67907899/223113793-4631ae6d-db6b-42a1-a9ea-ccf597e52613.png) | ![Dendrogram at 55.5](https://user-images.githubusercontent.com/67907899/223113852-845c6c58-1c92-4219-9521-1ed992683a64.png)

Graph (2): Dendrogram Presentation with Cutting Threshold
</div>

To evaluate the effectiveness of hierarchical clustering with Ward linkage, we performed clustering on different scenarios, including clustering before and after PCA with three and four clusters. Our analysis shows that the best results were obtained using hierarchical clustering with Ward linkage after applying PCA, as shown in Graph (3), and we received the maximum Silhouette score when utilizing four clusters.

<div align="center">

![Hierarchical Clustering](https://user-images.githubusercontent.com/67907899/223118352-ed60b3a1-4a3b-4285-a972-72886ec76228.png)

Graph (3): Hierarchical Clustering with Ward Linkage after PCA
</div>

## Cluster 1 Insights - Unemployed Travelers
Individuals in cluster 1 travel for 5 days on average, indicating a steady travel schedule. Interestingly, the majority of people in this cluster do not have a work, which might explain their travel habits. They use trip planning applications 2-3 to 6-7 days per week, indicating that they rely substantially on these apps for transportation. Also, they rarely use smartphone-app transportation services, with less than three days per month. It is worth mentioning that the majority of people in this group do not have a driver's license, which may explain their reliance on trip planning applications. When it comes to ride services, the majority of people in this cluster prefer Uber, with only a handful choosing Lyft or none at all. Additionally, because they are jobless, the majority of people in this cluster skipped logic relating to job type, work mode, and industry.
## Cluster 2 insights - Retired Errand Runners
Individuals in cluster 2 tend to travel only once during the week, likely for errands or appointments. This cluster includes individuals aged between 55 to 74 who do not have a job. Interestingly, individuals in this cluster rarely use trip-planning apps, with less than 3 days of usage per month. Moreover, information about their frequency of smartphone-app ride services usage and teleworking is missing, as many individuals skipped these questions. None of the individuals in this cluster are employed, and the majority do not have a driver's license. Additionally, they do not use either Uber or Lyft for their ride-sharing needs. Already many people in this cluster ignored questions regarding their job type, work style, industry, the purpose of their travels, and method of transportation for using smartphone-app ride services, indicating that their data was incomplete.
## Cluster 3 insights - Inconsistent Commuters
People in cluster 3 tend to travel for one to five full weekdays, showing some variation in their travel routine. Individuals in this cluster range in age from 35 to 64 and work only one job. Surprisingly, the frequency of smartphone-app transportation service usage is lacking in this cluster, since many people missed this question. Furthermore, none of the people in this cluster works from home. The majority of people in this cluster are working full-time and have a driver's license. They do not, however, use Uber or Lyft for their ride-sharing needs. The majority of them work in their actual workplace, but the reason for their trips is also missing, as many people skipped this question. Furthermore, the method of transportation for their smartphone-app ride services usage is absent, indicating that the data is incomplete.
## Cluster 4 insights - Working Commuters 
People in cluster 4 travel for 5 complete weekdays, indicating a steady travel pattern. Individuals in this cluster range in age from 25 to 54 and work only one job. They use trip planning applications 6-7 days each week, showing that they rely on them for transportation. Surprisingly, most people in this group rarely use smartphone-app transportation services, with less than three days per month. Furthermore, none of the people in this cluster work from home. The majority of people in this cluster are working full-time and have a driver's license. When it comes to ride-sharing services, the vast majority of people in this group prefer Uber. Additionally, the majority of them work in their actual workplace.

The analyzed data reveals interesting insights about each cluster's travel behaviour. The size of each cluster is shown in Graph (4).

<div align="center">

![size of each cluster with hierarchical clustering](https://user-images.githubusercontent.com/67907899/223119544-d9d53d2a-fcd7-43d9-8376-77b6fe96a3ae.png)

Graph (4): The Hierarchical Clustering Size of each Cluster
</div>

# K-Means Clustering
In order to find the optimal number of clusters for our data, we experimented with various values of k and different dimensions of data both before and after performing PCA. After careful consideration, we selected a model with k=4 on 2 dimensions as the best fit, as shown in Graph (5). We evaluated the models using the silhouette score as our metric of choice.

<div align="center">

![Screenshot 2023-03-06 153503](https://user-images.githubusercontent.com/67907899/223127490-b05347cf-7e8b-4a28-a2d9-b4f2becf2e32.png)

Graph (5): K-Means Clustering on 2-Dimensional Data. 
</div>

## Cluster 1 insights - Unemployed Travelers
The insights about cluster 1 remain largely consistent with the previous model. Individuals in cluster 1 still tend to travel 5 complete days, and most do not have a job. They use planning apps 2-3 and 6-7 days a week, and rarely use smartphone app ride services, with less than 3 days of usage per month. Similar to the previous model, individuals in this cluster do not have a driver's license, and many skipped questions related to job type, work mode, and industry. However, there is no information about telework frequency in this model.
## Cluster 2 insights - Full-time Working Commuters
The insights about cluster 2 suggest that individuals in this cluster tend to travel 5 complete weekdays, and are typically between the ages of 25 and 54. They have one job and use trip planning apps 6-7 days a week. Like individuals in cluster 1, they rarely use smartphone app ride services, with less than 3 days of usage per month. Most individuals in this cluster are employed full-time and have a driver's license.
## Cluster 3 insights - Flexible Commuters
Individuals in cluster 3 tend to travel either for one or five complete weekdays, indicating a flexible travel pattern. This cluster includes individuals aged between 35 to 64 who have one job. The frequency of their smartphone-app ride services usage is missing due to skipped logic, and most of them do not engage in teleworking. Most of the individuals in this cluster are employed full-time and have a driver's license. Interestingly, none of the individuals in this cluster uses either Uber or Lyft for their ride-sharing needs. Most of them work at their physical work location, but the purpose of their trips is missing due to skipped logic. Additionally, the mode of transportation for their smartphone-app ride services usage is missing, indicating incomplete data.
## Cluster 4 insights - Weekday Errand Runners
Individuals in cluster 4 tend to travel for one complete weekday, indicating a specific travel pattern. This cluster is characterized by individuals aged between 55 to 74 who are without a job. They rarely use trip-planning apps (less than 3 days a month) and have missing data on the frequency of their smartphone-app ride services usage and teleworking behaviour. Most of the individuals in this cluster are not employed and do not have a driver's license. Interestingly, none of the individuals in this cluster uses either Uber or Lyft for their ride-sharing needs. Data on job type, work mode, the purpose of their trips, industry, and the mode of transportation for smartphone-app ride services usage is missing due to skipped logic.

The data analysis uncovers intriguing insights about each cluster's travel pattern. Graph represents the size of each cluster (6).

<div align="center">

![k-mean clustering](https://user-images.githubusercontent.com/67907899/223124293-af89b38c-7fe1-4e9d-8f57-85e331528191.png)

Graph (6): Number of samples in each cluster after performing k means.
</div>

# DBSCAN Model
In order to conduct the DBSCAN algorithm, we performed hyperparameter tuning by testing various values of eps and minPts. We also tested the model on different datasets with varying dimensions, both before and after applying PCA. Our evaluation metric for determining the effectiveness of the model was the silhouette score, and based on this metric, we arrived at 6 clusters, as shown in Graph (7).

<div align="center">

![Screenshot 2023-03-06 153922](https://user-images.githubusercontent.com/67907899/223127437-f5bcf33a-dbde-49c3-9e65-e880338bae28.png)

Graph(7): DBSCAN Clustering on 2-Dimension
</div>

## Cluster 1 insight - Limited Mobility Commuters
Cluster 1 consists of individuals who are not currently employed and do not have a job. They are aged between 55 to 74 years old and tend to travel only one complete day per week. Most of them do not use smartphone app ride services (less than 3 days a month) and do not have a license. It seems that many individuals in this cluster skipped the logic in the job type and work mode questions since they are not currently employed. Additionally, they do not use Uber or Lyft services and the most common purpose of trips made using smartphone-app ride services is not applicable, given that they don't use these services.
## Cluster 2 insights	
This cluster seems to represent employed individuals in the age range of 25-54 who work full-time on work locations for five complete weekdays. They use TNC services (Uber and Lyft) occasionally (less than three days a month). Additionally, some individuals in this cluster also use rail or taxi services for transportation.
## Cluster 3 insights
The cluster's age range is from 25 to 64, and they tend to work either one or five complete weekdays. This suggests that they may have a full-time job, which could be why they primarily work on their work location and do not use ride-hailing services frequently.
## Cluster 4 insights
Cluster 4 seems to be made up of individuals who are not employed or currently looking for work. They have no specified job type or work mode. They also do not frequently use telework, indicating that they are not engaged in remote work or freelancing.
Interestingly, despite not being employed, many of the individuals in this cluster still use ride-hailing services such as Uber and Lyft, and some also use any rail and taxi services. This may suggest that they use these services for personal transportation needs rather than for work-related purposes.
## Cluster 5 insights
This cluster consists of people who are not currently employed and have not specified their job type or work mode. They rarely use TNCs (less than 3 days a month) but do use Uber and Lyft and usually travel on a weekday. The age range of this cluster is 55-64 years old, indicating that they are likely to be retirees or individuals who have left the workforce.
A significant portion of this cluster has a walking disability, which could impact their mobility options. Some individuals in this cluster use any rail, indicating that public transportation is still a mode of transportation for them despite their TNC usage.
Cluster 6 insights
People in this cluster aged between 55 to 64, they’re employees with one job and have 1 complete weekday, and they don’t use Uber or Lyft. 

The data analysis provides valuable insights about each cluster's travel behavior. The size of each cluster is displayed in Graph (8).

<div align="center">

![image](https://user-images.githubusercontent.com/67907899/223127852-4477ba5c-0441-418d-80cd-d9e9ca5bbc08.png)

Graph (8): Number of samples in each cluster after performing DBSCAN model. 
</div>
