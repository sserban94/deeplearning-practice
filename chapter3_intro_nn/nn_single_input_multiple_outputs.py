def neural_network(nn_input, weights):
    # In this case - one input multiple outputs
    # return an array of predictions
    prediction = compute_elementwise_multiplication(nn_input, weights)
    return prediction


def compute_elementwise_multiplication(element, array):
    output = [0, 0, 0]
    for i in range(len(array)):
        output[i] += array[i] * element
    return output


weights = [0.3, 0.2, 0.9]
win_loss_ratio = [0.65, 0.8, 0.8, 0.9]
nn_input = win_loss_ratio[0]
prediction = neural_network(nn_input, weights)
print(prediction)
