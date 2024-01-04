import numpy as np

'''
Notation reference 

x: Input sequence
y: Output sequence

x_t: Input at time t
h_t: Hidden state at time t
y_t: Output at time t

W_hh: Weights for the hidden layer
W_xh: Weights for the input layer
W_hy: Weights for the output layer

'''

def ReLU(x):
    return np.maximum(0, x)
def sigmoid(x):
    return (1/(1+np.exp(-x)))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def initialize_rnn_params(input_size, hidden_size, output_size):
    """

    Initialize weights for a simple RNN

    Args:
        input_size (int): Size of the input layer
        hidden_size (int): Size of the hidden layer
        output_size (int): Size of the output layer

    Returns:
        dict: A dictionary containing the randomly initialized weights and biases.

    """

    params = {
        'W_xh': np.random.randn(hidden_size, input_size) * 0.01,  # weights for input to hidden layer
        'W_hh': np.random.randn(hidden_size, hidden_size) * 0.01,  # weights for hidden to hidden layer
        'W_hy': np.random.randn(output_size, hidden_size) * 0.01,  # weights for hidden to output layer
        'b_h': np.random.randn(hidden_size) * 0.01,  # bias for hidden layer
        'b_y': np.random.randn(output_size) * 0.01  # bias for output layer
    }

    return params

def forward_pass(x_t, previous_hidden_state, params):
    """

    Args:
        x_t (numpy ndarray): Input feature vector at timestep t
        previous_hidden_state (numpy ndarray): Hidden state of the previous time step
        params (dict): Dictionary containing the trainable parameters

    Returns:
        h_t (numpy ndarray): The updated hidden state
        y_pred_t (numpy ndarray): The predicted output at current timestep

    """
    #Extract parameters

    W_xh = params['W_xh']
    W_hh = params['W_hh']
    W_hy = params['W_hy']
    b_h = params['b_h']
    b_y = params['b_y']

    #Compute the new hidden state
    a_t = np.dot(W_xh, x_t) + np.dot(W_hh, previous_hidden_state) + b_h
    h_t = tanh(a_t)

    # Compute the predicted output
    y_pred_t = np.dot(W_hy, h_t) + b_y

    return h_t, y_pred_t

def loss(y_true, y_pred):
    """
    Compute binary cross-entropy loss.

    Args:
        y_true (numpy ndarray): Array of true labels (0 or 1).
        y_pred (numpy ndarray): Array of predicted probabilities.

    Returns:
        float: Binary cross-entropy loss.
    """
    # Avoid division by zero and log(0) by clipping predictions
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Compute binary cross-entropy loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def backward_prop():
    pass

def update_params(params, gradients):

    W_xh = params['W_xh']
    W_hh = params['W_hh']
    W_hy = params['W_hy']
    b_h = params['b_h']
    b_y = params['b_y']

    dW_xh = gradients['dW_xh']
    dW_hh = gradients['dW_hh']
    dW_hy = gradients['dW_hy']
    db_h = gradients['db_h']
    db_y = gradients['db_y']

    

