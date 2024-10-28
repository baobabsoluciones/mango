from typing import List

import numpy as np


def calculate_network_output(
    x: np.array,
    weights: np.array,
    layers: int,
    input_nodes: int,
    hidden_nodes: List[int],
    output_nodes: int,
) -> np.array:
    """
    The calculate_network_output function takes in a set of inputs, weights, and nodes.
    It then calculates the output of the network by multiplying each input with its corresponding weight
    and adding it to a bias term for that layer. The function then passes this value through an activation function
    (sigmoid) and returns the result.

    :param x: the input data
    :type x: :class:`np.array`
    :param weights: the weights of each layer
    :type weights: :class:`np.array`
    :param int layers: the number of layers of the network
    :param int input_nodes: the number of input nodes
    :param list hidden_nodes: the list with the number of nodes in each layer
    :param int output_nodes: the number of output nodes
    :return: the result of the network for a given set of inputs
    :rtype: :class:`np.array`
    """
    start = 0
    for layer in range(layers + 1):
        if layer == 0:
            shape = (input_nodes, hidden_nodes[layer])
        elif layer == layers:
            shape = (hidden_nodes[-1], output_nodes)
        else:
            shape = (hidden_nodes[layer - 1], hidden_nodes[layer])
        end = start + shape[0] * shape[1]
        w = np.array(weights[start:end]).reshape(shape)
        start = end
        end = end + shape[1]
        bias = np.array(weights[start:end]).reshape((shape[1],))
        start = end

        x = x @ w + bias
        x = 1 / (1 + np.exp(-x))

    return x
