# Heart Disease Classification using Support Vector Machine (SVM)
This project demonstrates a classification task on the Heart Disease dataset using a Support Vector Machine (SVM) model with a radial basis function (RBF) kernel. The aim is to predict whether a patient has heart disease based on various medical attributes.

Dataset
The dataset used in this project is heart.csv. The target variable is output, which indicates the presence of heart disease (1 for presence, 0 for absence). The dataset is split into features (X) and the target label (y).

# Model Training
Model Used: Support Vector Machine (SVM)
Kernel: Radial Basis Function (RBF)
C: 20.00
Gamma: 0.01
Degree: 3 (default for RBF kernel)
The model is trained on 80% of the dataset and tested on the remaining 20%.

# Results
Accuracy


Train Accuracy: 100.0000 %


Test Accuracy: 97.0732 %

# Classification Report

              precision    recall  f1-score   support

           0       0.95      1.00      0.97       106
           1       1.00      0.94      0.97        99

           
   accuracy                            0.97       205

   
  macro avg        0.97      0.97      0.97       205

  
weighted avg       0.97      0.97      0.97       205

## Confusion Matrix

  [[106   0]

  
  [  6  93]]

## Visualization
You can visualize the confusion matrix and class distribution either interactively using Plotly or statically using Matplotlib.

Interactive Visualizations:

Confusion Matrix
Class Distribution
Static Visualizations:

Confusion Matrix
Class Distribution
To toggle between interactive and static visualizations, set the interactive variable in the code.

# Requirements


Python 3.x


Libraries:


numpy


pandas


matplotlib


seaborn


plotly


scikit-learn

## Install the required libraries using:

bash
Copy code

 ## pip install numpy pandas matplotlib seaborn plotly scikit-learn



## Usage

## Clone the repository.
1.Ensure you have the necessary libraries installed.


2.Place the heart.csv dataset in the working directory.


3.Run the script to train the model and visualize the results.


## This README should provide a clear overview of the project and instructions for users who want to replicate the work or build upon it.



