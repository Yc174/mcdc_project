import numpy as np

parameters = {
    'W1': [[ 1.6489735, 2.5770068,  1.3255011,  1.5453409, -1.7536916, -2.1271243, 3.5497422, -1.1388757,  1.4115595, -1.240432 ]],
    'b1': [[-2.3562818, -2.1307294, -1.9104774, -5.0135794, -2.2701466, -1.8844584, -1.8767567, -2.7430067, -3.9331024, -2.3172555]],
    'W2': [[3.4655418],
           [3.4225938],
           [3.4731112],
           [3.8253808],
           [0.13520782],
           [-0.2229167],
           [3.2454066],
           [0.39814854],
           [3.5073137],
           [0.34731576]],
    'b2' : [[1.9144869]],

}

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    
    return s


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    
    W1 = np.asarray(parameters['W1'])
    b1 = np.asarray(parameters['b1'])
    W2 = np.asarray(parameters['W2'])
    b2 = np.asarray(parameters['b2'])

    # Implement Forward Propagation to calculate A2 (probabilities)

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2

    # assert (Z2.shape == (1, X.shape[1]))

    return Z2