import numpy as np
import pandas as pd

def get_x_roof(X):
  big_sigma = X.sum()

  nn = X.size
  x_roof = (1 / nn) * big_sigma
  return x_roof

def get_little_sigma(X, x_roof=None):
  if x_roof is None:
    x_roof = get_x_roof(X)
  
  inner_power = X - x_roof
  inner_big_sigma = np.power(inner_power, 2)
  big_sigma = inner_big_sigma.sum()

  nn = X.size
  inner_root = (1 / nn) * big_sigma
  little_sigma = np.sqrt(inner_root)
  return little_sigma

def preprocess(X,y):
    """
    Perform Standardization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    
    x_roof = get_x_roof(X)
    little_sigma = get_little_sigma(X, x_roof=x_roof)
    
    X = (X - x_roof) / little_sigma
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    
    X = np.c_[np.ones(X.shape[0]), X]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_loss(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the loss.
    ###########################################################################
    # TODO: Implement the MSE loss function.                                  #
    ###########################################################################
    
    X_T = X.T
    predictions = np.dot(theta, X_T)
    errors = predictions - y
    inner_big_sigma = np.power(errors, 2)
    big_sigma = inner_big_sigma.sum()

    n = X.shape[0]
    J = (1 / (2 * n)) * big_sigma  
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    for i in range(num_iters) :
        y_pred = X @ theta # compute the predicted values (its vector)
        pred_error = (y_pred - y) # each entry in the vector equivalent to (thetaT - yi)
        inner_big_sigma = X.T @ pred_error

        # because on each entry in the gradient we should multiply the pred_error in the feature of the instance
        # respected to the place of the partial derivative
        # so we will use X transpose and aftet multiply X^t in pred_error we will get vector that each entry is
        # the partial derivative withoud divide it in n
        n = X.shape[0]
        gradient = inner_big_sigma / n
        theta -= (eta * gradient)  # update parameters

        J = compute_loss(X ,y ,theta)  # compute loss (MSE)
        J_history.append(J)  # store loss value
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    
    XT = X.T
    XT_X = XT @ X
    XT_X_inv = np.linalg.inv(XT_X)  # If XT_X is invertible
    pinv_theta = XT_X_inv @ X.T @ y
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than epsilon. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent with stop condition optimization algorithm.  #
    ###########################################################################
    
    for i in range(max_iter):
        theta, J = gradient_descent(X, y, theta, eta, 1)
        J_history += J  # store loss value
        if (i > 1) and (abs(J_history[-1] - J_history[-2]) < epsilon):
            break
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def efficient_gradient_descent(*args, **kwargs):
    return gradient_descent_stop_condition(*args, **kwargs)

def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using 
    the training dataset. Maintain a python dictionary with eta as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """
    
    etas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    eta_dict = {} # {eta_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################
    
    theta = np.random.random(size=2)
    for eta in etas:
        new_theta, J_history = gradient_descent(X_train, y_train, theta, eta, iterations)

        J = compute_loss(X_val, y_val, new_theta)
        eta_dict[eta] = J
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return eta_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################
    
    X_train = apply_bias_trick(X_train)
    X_val = apply_bias_trick(X_val)
    shape = X_train.shape[1]
    
    max_features = 5
    features_list = list(range(1, X_train.shape[1]))
    
    for i in range(max_features):
        losses = {}
        theta = np.random.random(len(selected_features) + 2)
        for j in features_list:
            M_temp = selected_features + [0, j, ]
            partly_X_train = X_train[:, M_temp]
            theta_test, J_history = efficient_gradient_descent(partly_X_train, y_train, theta, best_eta, iterations)
            partly_X_val = X_val[:, M_temp]
            losses[j] = compute_loss(partly_X_val, y_val, theta_test)
        
        best_j = min(losses, key=losses.get)
        selected_features.append(best_j - 1)
        features_list.remove(best_j)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    for col in df_poly:
        df_poly[f"{col}^2"] = df[col] ** 2
    
    columns = df_poly.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            df_poly[f"{col1}*{col2}"] = df[col1] * df[col2]
    
    #########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly