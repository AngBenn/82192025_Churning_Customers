# 82192025_Churning_Customers

Customer Churn Prediction
Overview
This project focuses on predicting customer churn using machine learning techniques. The dataset used in this project contains information about customers, including features such as gender, tenure, contract type, and more.

Contents
Data Exploration and Visualization:

Explored the dataset and visualized relationships between different features and customer churn.
Data Preprocessing:

Removed unnecessary columns (e.g., 'customerID').
Encoded categorical columns using LabelEncoder.
Random Forest Model:

Trained a Random Forest Classifier to predict customer churn.
Selected important features using feature importance.
Keras Model:

Built a Keras neural network for customer churn prediction.
Conducted hyperparameter tuning using GridSearchCV.
Model Evaluation:

Evaluated models on accuracy and AUC score.
Retrained Model:

Retrained the Keras model with the best parameters on the entire training dataset.
Model Save:

Saved the best Keras model and the associated scaler using pickle.
Files
CustomerChurn_model.pkl: Pickle file containing the best Keras model.
scaler.pkl: Pickle file containing the scaler used for standardization.
Usage
Data Preparation:

Ensure that the dataset (CustomerChurn_dataset.csv) is available.
Mount Google Drive using the provided code.
Exploration and Preprocessing:

Run the code for data exploration and preprocessing.
Random Forest Model:

Train and evaluate the Random Forest model.
Keras Model:

Train and evaluate the Keras neural network.
Save the best Keras model and scaler.
Retrained Model:

Retrain the Keras model on the entire training dataset.
Evaluate the retrained model on the testing set.
Dependencies
pandas
numpy
tensorflow
scikit-learn
seaborn
matplotlib
scikeras
Acknowledgments
This project was developed as part of a customer churn prediction task. Feel free to adapt and extend it based on your specific requirements.
