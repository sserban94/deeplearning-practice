# toe #%win #fans
input_weights = [
    [0.1, 0.2, -0.1],  # hidden_weights[0]?
    [-0.1, 0.1, 0.9],  # hidden_weights[1]?
    [0.1, 0.4, 0.1],  # hidden_weights[2]?
]
# the hidden weights are the outputs from the first prediction
# hidden as I don't know them from the beginning
# they will be computed

#   h[0] #h[1] #h[2]
hidden_weights = [
    [0.3, 1.1, -0.3],  # hurt?
    [0.1, 0.2, 0.0],  # win?
    [0.0, 1.3, 0.1],  # sad?
]

weights = [input_weights, hidden_weights]

average_toe_number = [8.5, 9.5, 9.9, 9.0]  # toe
win_loss_ratio = [0.65, 0.8, 0.8, 0.9]  # %win
average_fan_number = [1.2, 1.3, 0.5, 1.0]  # fans

nn_input = [average_toe_number[0], win_loss_ratio[0], average_fan_number[0]]


def compute_dot_product(firstArray, secondArray):
    if len(firstArray) != len(secondArray):
        return
    dot_product = 0
    for i in range(len(firstArray)):
        dot_product += firstArray[i] * secondArray[i]
    return dot_product


def compute_array_matrix_multiplication(nn_input, weights):
    if len(nn_input) != len(weights):
        return
    output = [0, 0, 0]
    for i in range(len(nn_input)):
        output[i] = compute_dot_product(nn_input, weights[i])
    return output

def neural_network(nn_input, weights):
    hidden_input = compute_array_matrix_multiplication(nn_input, weights[0])
    prediction = compute_array_matrix_multiplication(hidden_input, weights[1])
    return prediction

prediction = neural_network(nn_input, weights)
print(prediction)