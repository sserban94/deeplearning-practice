# toe #%win #fans
weights = [
    [0.1, 0.1, -0.3],  # hurt?
    [0.1, 0.2, 0.0],  # win?
    [0.0, 1.3, 0.1],  # sad?

]

average_toe_number = [8.5, 9.5, 9.9, 9.0]  # toe
win_loss_ratio = [0.65, 0.8, 0.8, 0.9]  # %win
average_fan_number = [1.2, 1.3, 0.5, 1.0]  # fans

nn_input = [average_toe_number[0], win_loss_ratio[0], average_fan_number[0]]


def compute_array_matrix_multiplication(array, matrix):
    if len(array) != len(matrix):
        return

    output = [0, 0, 0]

    for i in range(len(array)):
        output[i] = compute_dot_product(array, matrix[i])

    return output


def compute_dot_product(array, matrixArray):
    if len(array) != len(matrixArray):
        return

    weighted_sum = 0

    for i in range(len(array)):
        weighted_sum += array[i] * matrixArray[i]

    return weighted_sum


def neural_network(nn_input, weights):
    prediction = compute_array_matrix_multiplication(nn_input, weights)
    return prediction

# Basically here I get three independent dot products
# one dot product for input[toes, win/loss, fans] with output - hurt
# one dot product for input[toes, win/loss, fans] with output - win
# one dot product for input[toes, win/loss, fans] with output - sad
prediction = neural_network(nn_input, weights)
print(prediction)

# The result will be:
# [0.555, 0.9800000000000001, 0.9650000000000001]
#   hurt P        win P               sad P

# For now - computing the weighted sums - vector * each row in a matrix
# Linear algebra tells us that weights should be processed as column vectors instead of rows