#Heart Disease Classification using Support Vector Machine (SVM)
Overview
Heart disease is a leading cause of death globally, and early detection is crucial for effective treatment. This project aims to build a predictive model that can classify whether a patient has heart disease based on various medical attributes. The model used for this task is a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel, which is known for its effectiveness in classification problems with non-linear decision boundaries.

Dataset
The dataset used in this project is heart.csv, which contains 13 features related to heart health, such as age, sex, chest pain type, resting blood pressure, cholesterol level, and more. The target variable is output, where:

1 indicates the presence of heart disease.
0 indicates the absence of heart disease.
The dataset is split into two sets:

Training Set: 80% of the data, used to train the SVM model.
Test Set: 20% of the data, used to evaluate the model's performance.
Model Selection
Support Vector Machine (SVM)
Support Vector Machines are powerful classification models that work by finding the optimal hyperplane that separates data points of different classes. The RBF kernel is particularly effective for this dataset as it maps the input features into a higher-dimensional space, allowing the model to handle non-linear relationships between features and the target variable.

Hyperparameters
C: 20.62 (Controls the trade-off between achieving a low training error and a low testing error)
Gamma: 0.01 (Defines how far the influence of a single training example reaches)
Kernel: RBF (Radial Basis Function)
Degree: 3 (Though this is more relevant for polynomial kernels, it’s set as the default value)
Model Training and Evaluation
The model was trained using the X_train and y_train datasets and evaluated on X_test and y_test. The performance metrics include accuracy, precision, recall, F1-score, and the confusion matrix.

Results
Train Accuracy: 100.0000 %
Test Accuracy: 97.0732 %
The high accuracy on both the training and test sets indicates that the model generalizes well to unseen data, suggesting that overfitting is not a concern with the selected hyperparameters.

Classification Report
The classification report provides a detailed breakdown of the model's performance on each class (0 for absence and 1 for presence of heart disease). The precision, recall, and F1-score are close to 1, indicating that the model performs exceptionally well on this dataset.

Confusion Matrix
The confusion matrix shows the number of correct and incorrect predictions for each class:

lua
Copy code
[[106   0]
 [  6  93]]
This matrix highlights the model's ability to correctly identify cases of heart disease, with only 6 misclassifications out of 205 predictions.

Visualization
To better understand the model's performance, two types of visualizations are provided:

Interactive Visualizations (Using Plotly)
Confusion Matrix: An interactive heatmap that displays the confusion matrix with color-coded counts for each class.
Class Distribution: An interactive bar chart that shows the distribution of predicted classes.
Static Visualizations (Using Matplotlib and Seaborn)
Confusion Matrix: A static heatmap similar to the interactive version, but rendered using Matplotlib and Seaborn.
Class Distribution: A static bar chart displaying the predicted class distribution.
Users can toggle between interactive and static visualizations by setting the interactive variable in the code.

Installation
To run this project, you need Python 3.x and the following Python libraries:

numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
You can install the required libraries using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn plotly scikit-learn
Usage
Clone the Repository: Download or clone the repository to your local machine.
Prepare the Dataset: Ensure that the heart.csv dataset is placed in the working directory.
Run the Script: Execute the script to train the model and generate visualizations.
bash
Copy code
python heart_disease_classification.py
Analyze Results: The script will output the model’s accuracy, classification report, confusion matrix, and visualizations.
Conclusion
This project demonstrates the effectiveness of SVM with an RBF kernel in classifying heart disease. With an impressive test accuracy of over 97%, the model is robust and reliable for predictive analysis in medical data. Future work could involve experimenting with different models, hyperparameter tuning, or expanding the dataset to include more features or samples.

License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as you see fit.

This extended description provides more context, detailed explanations, and guides the user through every aspect of the project. It also includes a section on installation, usage, and possible future improvements.






