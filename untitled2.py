# -*- coding: utf-8 -*-
Original file is located at
    https://colab.research.google.com/drive/1596SyleOyL3prOZ44whWCrmpGT1ukk19
"""

historical_data = [0, 1, 6, 7, 12, 12, 17, 22, 21, 16, 5, 10, 15, 20, 21,15,20,14,1,9,13,0]

import numpy as np

# Initialize a 25x25 transition matrix with small values to avoid zero probabilities
transition_matrix = np.ones((25, 25)) * 1e-5

def update_transition_matrix(data, transition_matrix):
    for i in range(len(data) - 1):
        current_state = data[i]
        next_state = data[i + 1]
        transition_matrix[current_state, next_state] += 1

    # Normalize rows to get probabilities
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

# Update transition matrix with historical data
update_transition_matrix(historical_data, transition_matrix)

def extract_features(data):
    features = []
    labels = []
    for i in range(len(data) - 1):
        features.append(data[i])  # Current position
        labels.append(data[i + 1])  # Next position
    return np.array(features).reshape(-1, 1), np.array(labels)

features, labels = extract_features(historical_data)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert labels to integers if they are not already
y_train = y_train.astype(str)  # Ensure labels are integers
y_test = y_test.astype(str)    # Ensure labels are integers

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

prior_prob = np.ones(25) / 25

def likelihood(observed_location, current_location):
    # Define your likelihood function based on the observed data
    return 1.0  # Example likelihood value, adjust based on your model

def update_posterior(prior, likelihood, observed_location):
    posterior = prior * likelihood(observed_location, np.arange(25))
    posterior /= posterior.sum()  # Normalize to make it a probability distribution
    return posterior

# Example update
observed_location = 1
posterior_prob = update_posterior(prior_prob, likelihood, observed_location)

# Assume equal weights for simplicity
weights = np.array([0.33, 0.33, 0.34])

# Example transition matrix probability for the current state
current_location = 9
markov_prob = transition_matrix[current_location]

# Example machine learning model prediction probability
ml_prob = np.zeros(25)
# Predict the location and convert it to an integer for indexing
predicted_index = int(model.predict([[current_location]])[0])
ml_prob[predicted_index] = 1.0


# Combine probabilities
combined_prob = weights[0] * markov_prob + weights[1] * ml_prob + weights[2] * posterior_prob

# Predict the next location
predicted_location = np.argmax(combined_prob)

print(f"Predicted next location: {predicted_location}")

