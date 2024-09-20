# Car Data Segmentation with Hierarchical Clustering

![Python](https://img.shields.io/badge/Python-3.8+-green)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Jupyter%20Notebook-orange)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-blue)
![Pandas](https://img.shields.io/badge/Library-Pandas-yellow)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-lightblue)


## Project Overview
This project focuses on Car Data Segmentation using Hierarchical Clustering, a method commonly used to group similar cars based on various attributes. This segmentation can be valuable for automakers, dealers, and customers to identify trends, preferences, and market segments. The primary objective is to analyze a dataset of car specifications and classify the cars into different segments based on their similarities.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Hierarchical Clustering](#hierarchical-clustering)
- [Results](#results)

## Introduction
Car segmentation helps identify distinct groups of cars based on characteristics such as engine size, horsepower, price, and fuel efficiency. This project employs Hierarchical Clustering, a popular unsupervised learning method, to classify cars into segments, giving insights into patterns and relationships within the data.

Hierarchical Clustering builds a hierarchy of clusters by successively merging or splitting existing clusters, allowing us to visualize the segmentation using a dendrogram.

## Features
- Car data segmentation using Agglomerative Hierarchical Clustering.
- Dendrogram visualization to explore hierarchical relationships between car segments.
- Data preprocessing to handle missing values, normalize data, and ensure effective clustering.
- Segment insights to identify key groupings of cars based on attributes.

## Dataset
Each row represents a unique car model, and each column describes a specific attribute of that car. The dataset includes a variety of features such as miles per gallon (mpg), cylinders (cyl), displacement (disp), and horsepower (hp), among others.

Here’s an overview of each attribute:

**Model**: The name of the car model.

**mpg (Miles per Gallon)**: This represents the fuel efficiency of the car, measured in miles that the car can travel per gallon of fuel. Higher values indicate better fuel efficiency.

**cyl (Number of Cylinders)**: Indicates the number of engine cylinders. Common values are 4, 6, or 8. Cars with more cylinders tend to be more powerful but less fuel-efficient.

**disp (Displacement)**: Engine displacement in cubic inches, which indicates the volume of the car's cylinders. A higher displacement generally means more power but lower fuel efficiency.

**hp (Horsepower)**: A measure of the car’s power output. Cars with higher horsepower can accelerate faster and reach higher speeds.

**drat (Rear Axle Ratio)**: The ratio of the car’s drivetrain, affecting the relationship between engine speed and wheel rotation. It impacts the car's performance, such as acceleration and fuel efficiency.

**wt (Weight)**: The weight of the car in tons. Heavier cars generally require more power to move and tend to be less fuel-efficient.

**qsec (Quarter-Mile Time)**: The time, in seconds, it takes for the car to complete a quarter-mile distance. Lower values indicate better acceleration.

**vs (Engine Shape)**: A binary variable where 0 indicates a V-shaped engine and 1 indicates a straight engine configuration.

**am (Transmission)**: Indicates the type of transmission: 0 for automatic and 1 for manual.

**gear (Number of Gears)**: The number of forward gears in the transmission system.

**carb (Number of Carburetors)**: Indicates the number of carburetors present in the car’s engine. More carburetors generally allow for more air and fuel to enter the engine, leading to higher performance but potentially lower fuel efficiency

dataset link : https://www.kaggle.com/datasets/muhammadyasirsaleem/car-model-dataset


## Data Preprocessing:


### Missing Values :

Missing data can significantly impact the accuracy and reliability of your analysis. Here are some common approaches to handle missing values:

Deletion:
Listwise deletion: Remove entire rows or columns containing missing values.

Pairwise deletion: Exclude pairs of observations with missing values for a specific variable.

Imputation:
Mean/median imputation: Replace missing values with the mean or median of the respective column.

Mode imputation: Replace missing values with the most frequent value in the column.

Regression imputation: Predict missing values using a regression model based on other variables.

Hot-deck imputation: Replace missing values with values from similar observations.

Multiple imputation: Create multiple complete datasets by imputing missing values in different ways.

### Outliers

Outliers are data points that deviate significantly from the rest of the data. They can have a disproportionate impact on your analysis, especially when using methods sensitive to outliers (e.g., mean, standard deviation). Here are some strategies to handle outliers:   

Identification:
Visual inspection: Use plots (e.g., box plots, histograms) to identify outliers visually.

Statistical methods: Calculate measures like z-scores or interquartile range (IQR) to identify outliers based on predefined thresholds.

Treatment:

Removal: Remove outliers if they are clearly erroneous or have a significant impact on your analysis.

Capping: Replace outliers with a maximum or minimum value to limit their influence.

Winsorization: Replace outliers with the nearest non-outlier value.

Transformation: Apply transformations (e.g., log transformation) to reduce the impact of outliers.

### Normalization and Scaling

Normalization and scaling are essential preprocessing steps to ensure that all features contribute equally to the analysis. This is particularly important when using distance-based algorithms (e.g., k-means clustering, principal component analysis).

Normalization:

Scales features to a specific range (e.g., 0 to 1).

Scaling:

Robust scaling: Scales features using the median and interquartile range to be less sensitive to outliers.

## Hierarchical Clustering

Hierarchical Clustering is a popular method of clustering in machine learning that builds a hierarchy of clusters. It is an unsupervised learning technique used to group data points with similar characteristics into clusters without prior knowledge of the number of clusters. The result is often presented as a dendrogram, a tree-like diagram that represents the hierarchy of clusters.

Agglomerative Hierarchical Clustering :

- Bottom-up approach: Starts with each data point as a separate cluster and merges clusters iteratively based on similarity.
  
- Similarity measures: Euclidean distance, Manhattan distance, cosine similarity, etc.
  
Linkage methods:

- Single-linkage: The distance between two clusters is defined as the minimum distance between any pair of points from the two clusters.

- Complete-linkage: The distance between two clusters is defined as the maximum distance between any pair of points from the two clusters.

- Average linkage: The distance between two clusters is defined as the average distance between all pairs of points from the two clusters.

- Centroid-linkage: The distance between two clusters is defined as the distance between the centroids of the two clusters. 

  Dendrogram:

 A dendrogram is generated to visualize the hierarchy of clusters, providing insights into the segmentation process.The number of clusters is chosen based on where the dendrogram is cut



## Results
After running the project, you'll observe:

Dendrogram: Visual representation of the hierarchical clustering process.

Clustered Data: A breakdown of cars into different clusters or segments based on the features.

### dendrogram
![dendorom](https://github.com/user-attachments/assets/e99a24f0-ef59-4260-b5fe-bf01f5b1fffe)

A horizontal shows the threshold from where the clusters can be identified. Currently there are 3 clusters shown.


### Adjusted Rand Index (ARI) :

The Adjusted Rand Index (ARI) or Adjusted Rand Score is a metric used to evaluate the similarity between two clusterings by comparing how closely they match.

The ARI is calculated based on the number of element pairs that are either assigned to the same or different clusters in both clusterings. It ranges from -1 to 1:

![Adjusted Rand Index](https://github.com/user-attachments/assets/4ff7232d-3d6d-4c2f-a477-e724366c2167)


### Contingency Table :

A contingency table, also known as a cross-tabulation or crosstab, is a statistical table that shows the frequency distribution of two or more categorical variables. It helps visualize and analyze the relationship between these variables.

![compersion](https://github.com/user-attachments/assets/d94ddcc8-8606-42b3-996b-1dca599244c1)

Rows and Columns:

The rows and columns are indexed by different categories or measurements. It appears that the rows are indexed as row_0, row_1, row_2, and the columns correspond to the number of cylinders (4, 6, 8).

In this table:

- row_0:
The values are [0,2,13], indicating that there are 2 occurrences for car engines with 6-cylinders, and 13 occurrences for car engine with 8-cylinders but none for 4 cylinders car.

- row_1:
The values are [11,5,0], indicating that there are 11 occurrences for engines with 4 cylinders and 5 occurrences for engines with 6 cylinders, but none for 8 cylinders.

- row_2:
The values are [0,0,1], indicating that there are 1 occurrence for car engines with 8 cylinders but none for 4 and 6 car cylinders.







