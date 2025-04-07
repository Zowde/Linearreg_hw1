Linear Regression with Gradient Descent

This project implements a linear regression model using gradient descent to predict continuous values. The model is trained using a dataset (data.csv) containing 5000 instances, with various functions to optimize and evaluate the performance of the model. Key features of the project include data preprocessing, applying the bias trick, loss computation, and gradient descent optimization.

Functions Overview
Data Preprocessing:
The preprocess() function standardizes the input features and labels using the z-score method.

Bias Trick:
The apply_bias_trick() function adds a column of ones to the feature set to account for the bias term in the model.

Loss Calculation:
The compute_loss() function computes the Mean Squared Error (MSE) between the predicted and actual values, used to measure model performance.

Gradient Descent:
The gradient_descent() function optimizes the model parameters by iteratively adjusting them to minimize the loss function.

Pseudoinverse:
The compute_pinv() function computes the optimal model parameters using the closed-form solution (pseudoinverse).

Gradient Descent with Stop Condition:
The gradient_descent_stop_condition() function implements an early stopping condition to halt training once the improvement in the loss value is minimal.

Learning Rate Optimization:
The find_best_learning_rate() function tests different learning rates to find the best one that minimizes validation loss.

Forward Feature Selection:
The forward_feature_selection() function selects the most important features through an iterative process to improve model accuracy.

Polynomial Features:
The create_square_features() function generates polynomial features (squared and interaction terms) to capture more complex relationships between the features.

Requirements
Python 3.x

NumPy

Pandas

Data
The project uses a dataset data.csv, which contains 5000 instances and several features. The data is preprocessed and standardized before training the model.

How to Run
Load the dataset using pandas.

Preprocess the data using the preprocess() function.

Apply the bias trick to the data with apply_bias_trick().

Train the model using gradient_descent() or efficient_gradient_descent().

Evaluate model performance using the compute_loss() function.

Optionally, perform forward feature selection or use polynomial features for enhanced accuracy.
