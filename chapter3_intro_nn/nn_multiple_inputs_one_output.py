def compute_weighted_sum(first_array, second_array):
    # Don't forget the check the lengths of the arrays/vectors
    if len(first_array) != len(second_array):
        return
    weighted_sum = 0
    for i in range(len(first_array)):
        # multiply each input with its corresponding weight and compute the weighted sum
        weighted_sum += first_array[i] * second_array[i]
    return weighted_sum


def neural_network(nn_input, weights):
    prediction = compute_weighted_sum(nn_input, weights)
    return prediction


weights = [0.1, 0.2, 0]

average_toe_number = [8.5, 9.5, 9.9, 9.0]
win_loss_ratio = [0.65, 0.8, 0.8, 0.9]
average_fan_number = [1.2, 1.3, 0.5, 1.0]

# Again - using only the first elements - in this case current status at the beginning of each game
# avg toe, win loss and avg fan num all from the first match in the series
nn_input = [average_toe_number[0], win_loss_ratio[0], average_fan_number[0]]
prediction = neural_network(nn_input, weights)
print(prediction)

# Weighted sum between two vectors (same len) = Dot Product
# any operations between two vectors with the same size pairing up values according to their index
#       = Elementwise operation
# In this case - Elementwise Multiplication

# IMPORTANT
# Why does a weighted sum / dot product work?
# Basically a dot product = notion of similarity between two vectors

# NN making a prediction = NN gives high score of the inputs based on their similarity with their weights
