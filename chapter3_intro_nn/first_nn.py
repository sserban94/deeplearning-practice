# Supervised Parametric Basic NN

# Paradigm - Predict Compare Learn

# Predict step ahead

# One input - one output

# most basic neural network ever
def neural_network(nn_input, weight):
    # multiplying prediction with weight ("knob")
    prediction = nn_input * weight
    return prediction


weight = 0.1

average_toe_number = [8.5, 9.5, 10, 9]
# Inserting one input datapoint
input = average_toe_number[0]  # only using the first element for now
prediction = neural_network(input, weight)
print(prediction)

# Info
# Input - number recorded in real world - something knowable - ex: temperature, yesterday's stock price
# Prediction - result computer by nn based on input
# Weight - the 'knob' which will be adjusted after each prediction - the nn learns from its mistakes
# The main idea - the neural network learns through trial and error

# Input = Information
# Weight = Knowledge
# Output = Prediction

# What's to remember from this example?
# The NN doesn't have any memory. If I send input[1] it has no clue about the prediction from input[0]
# At least for now
# Later on I will learn about short term memory

# Remember
# Weight = measure of sensitivity between input and prediction
# If weight high => even tiny input can create large prediction and vice versa
