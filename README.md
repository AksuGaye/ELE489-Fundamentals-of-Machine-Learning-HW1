# K-NN Algorithm on Wine Dataset

## Overview
This project is made for Homework of ELE489: Fundamentals of Machine Learning course. 
In this project, the k-Nearest Neighbors (K-NN) algorithm was implemented from scratch and applied to the Wine dataset from the UCI Machine Learning Repository. The dataset consists of 178 instances and 13 numerical features with class labels (1, 2, or 3). The goal was to compare the performance of the K-NN algorithm using various distance calculations which are Euclidean, Manhattan, and Minkowski and analyze the impact of the parameter K on classification accuracy.

## Dataset

The Wine dataset used in this project can be found at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine). It includes the following features:

- Alcohol
- Malic Acid
- Ash
- Alcalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315 of Diluted Wines
- Proline

The target variable is the Class (1, 2, or 3), representing different wine categories.

## Contents

- **Preprocessing**: Data normalization was performed to scale the features and ensure the k-NN algorithm works efficiently.
- **Visualization**: Various plots were created to visualize feature distributions and class distributions in the training and test sets.
- **k-NN Algorithm**: Implementing the k-NN algorithm from scratch, allowing for the calculation of different distances between the test point and training points.

## Distance Metrics

1. **Euclidean Distance**: Measures the straight-line distance between two points in the feature space.
2. **Manhattan Distance**: Measures the sum of absolute differences along each dimension.
3. **Minkowski Distance**: A generalization of both Euclidean and Manhattan distances, with the parameter p controlling the metric.
   
## Results

-The k-NN algorithm was evaluated using different K values (1, 3, 5, 7, 9).
-Accuracy was calculated for each distance metric and K value, and performance was compared.
-Confusion Matrices and Classification Reports were generated for a deeper understanding of the model's performance for each distance metric.

## Conculsion

Manhattan Distance consistently performed the best, especially for K = 5, with accuracy reaching approximately 0.95. 

## Code Files

1. **knn.py**: This file contains the implementation of the k-NN algorithm from scratch. It includes functions for calculating distances, finding nearest neighbors, and making predictions based on the majority class in the neighborhood.
2. **analysis.ipynb**: A Jupyter notebook containing code, visualizations, and detailed explanations of the model's performance using various metrics.
3. **README.md**: This file provides an overview of the project, dataset, and how to run the code.

## Instructions to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AksuGaye/ELE489-Fundamentals-of-Machine-Learning-HW1.git

