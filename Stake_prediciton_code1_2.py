{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPC5rn+rISMy9nKSZWV2NQu",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/aryan-py/Stake_mine_predicitor/blob/main/Stake_prediciton_code1_2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zDUHTg69h5n1",
        "outputId": "30c2de75-4a4f-48b9-8292-b596d4691cfc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/50\n",
            "10/10 [==============================] - 1s 2ms/step - loss: 3.2343 - accuracy: 0.0000e+00\n",
            "Epoch 2/50\n",
            "10/10 [==============================] - 0s 2ms/step - loss: 3.0867 - accuracy: 0.3000\n",
            "Epoch 3/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 2.9634 - accuracy: 0.6000\n",
            "Epoch 4/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.8307 - accuracy: 0.8000\n",
            "Epoch 5/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.6701 - accuracy: 0.9000\n",
            "Epoch 6/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.4869 - accuracy: 1.0000\n",
            "Epoch 7/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.2739 - accuracy: 1.0000\n",
            "Epoch 8/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.0428 - accuracy: 1.0000\n",
            "Epoch 9/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 1.8020 - accuracy: 1.0000\n",
            "Epoch 10/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 1.5435 - accuracy: 1.0000\n",
            "Epoch 11/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 1.2978 - accuracy: 1.0000\n",
            "Epoch 12/50\n",
            "10/10 [==============================] - 0s 5ms/step - loss: 1.0637 - accuracy: 1.0000\n",
            "Epoch 13/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.8676 - accuracy: 1.0000\n",
            "Epoch 14/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.6950 - accuracy: 1.0000\n",
            "Epoch 15/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.5451 - accuracy: 1.0000\n",
            "Epoch 16/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.4228 - accuracy: 1.0000\n",
            "Epoch 17/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.3297 - accuracy: 1.0000\n",
            "Epoch 18/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.2596 - accuracy: 1.0000\n",
            "Epoch 19/50\n",
            "10/10 [==============================] - 0s 5ms/step - loss: 0.2081 - accuracy: 1.0000\n",
            "Epoch 20/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.1694 - accuracy: 1.0000\n",
            "Epoch 21/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.1378 - accuracy: 1.0000\n",
            "Epoch 22/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.1159 - accuracy: 1.0000\n",
            "Epoch 23/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0980 - accuracy: 1.0000\n",
            "Epoch 24/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0850 - accuracy: 1.0000\n",
            "Epoch 25/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0741 - accuracy: 1.0000\n",
            "Epoch 26/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0651 - accuracy: 1.0000\n",
            "Epoch 27/50\n",
            "10/10 [==============================] - 0s 5ms/step - loss: 0.0573 - accuracy: 1.0000\n",
            "Epoch 28/50\n",
            "10/10 [==============================] - 0s 5ms/step - loss: 0.0514 - accuracy: 1.0000\n",
            "Epoch 29/50\n",
            "10/10 [==============================] - 0s 5ms/step - loss: 0.0463 - accuracy: 1.0000\n",
            "Epoch 30/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0419 - accuracy: 1.0000\n",
            "Epoch 31/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0381 - accuracy: 1.0000\n",
            "Epoch 32/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0348 - accuracy: 1.0000\n",
            "Epoch 33/50\n",
            "10/10 [==============================] - 0s 6ms/step - loss: 0.0322 - accuracy: 1.0000\n",
            "Epoch 34/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0296 - accuracy: 1.0000\n",
            "Epoch 35/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0273 - accuracy: 1.0000\n",
            "Epoch 36/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0254 - accuracy: 1.0000\n",
            "Epoch 37/50\n",
            "10/10 [==============================] - 0s 7ms/step - loss: 0.0236 - accuracy: 1.0000\n",
            "Epoch 38/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0220 - accuracy: 1.0000\n",
            "Epoch 39/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0206 - accuracy: 1.0000\n",
            "Epoch 40/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0194 - accuracy: 1.0000\n",
            "Epoch 41/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0182 - accuracy: 1.0000\n",
            "Epoch 42/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0171 - accuracy: 1.0000\n",
            "Epoch 43/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0162 - accuracy: 1.0000\n",
            "Epoch 44/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0153 - accuracy: 1.0000\n",
            "Epoch 45/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0145 - accuracy: 1.0000\n",
            "Epoch 46/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0137 - accuracy: 1.0000\n",
            "Epoch 47/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0131 - accuracy: 1.0000\n",
            "Epoch 48/50\n",
            "10/10 [==============================] - 0s 5ms/step - loss: 0.0124 - accuracy: 1.0000\n",
            "Epoch 49/50\n",
            "10/10 [==============================] - 0s 15ms/step - loss: 0.0118 - accuracy: 1.0000\n",
            "Epoch 50/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0113 - accuracy: 1.0000\n",
            "1/1 [==============================] - 0s 200ms/step\n",
            "Predicted bomb location: (1, 4)\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Flatten\n",
        "from tensorflow.keras.utils import to_categorical\n",
        "\n",
        "# Generate dummy data for previous bomb locations (as a list of tuples)\n",
        "# Example: [(2, 3), (0, 1), (4, 4), ...]\n",
        "# Convert the list of tuples to a numpy array\n",
        "previous_bomb_locations = np.array([\n",
        "    [2, 3], [0, 1], [4, 4], [2, 2], [3, 0],\n",
        "    [1, 1], [0, 0], [4, 1], [3, 3], [1, 4]\n",
        "])\n",
        "\n",
        "# Convert the bomb locations to one-hot encoding\n",
        "def convert_to_one_hot(locations, grid_size=5):\n",
        "    one_hot = np.zeros((len(locations), grid_size, grid_size))\n",
        "    for i, (x, y) in enumerate(locations):\n",
        "        one_hot[i, x, y] = 1\n",
        "    return one_hot\n",
        "\n",
        "# Prepare the input data and labels\n",
        "X = convert_to_one_hot(previous_bomb_locations)  # Shape: (n_samples, 5, 5)\n",
        "y = convert_to_one_hot(previous_bomb_locations)  # Labels are same as inputs for simplicity\n",
        "\n",
        "# Reshape the input data to match the expected input shape for the neural network\n",
        "X = X.reshape((X.shape[0], 5, 5, 1))\n",
        "\n",
        "# Define the neural network model\n",
        "model = Sequential([\n",
        "    Flatten(input_shape=(5, 5, 1)),\n",
        "    Dense(128, activation='relu'),\n",
        "    Dense(128, activation='relu'),\n",
        "    Dense(25, activation='softmax')  # 25 units for 25 possible locations\n",
        "])\n",
        "\n",
        "# Compile the model\n",
        "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "# Train the model\n",
        "model.fit(X, y.reshape((y.shape[0], 25)), epochs=50, batch_size=1)\n",
        "\n",
        "# Function to predict the bomb location\n",
        "def predict_bomb_location(model, location):\n",
        "    location_one_hot = convert_to_one_hot([location]).reshape((1, 5, 5, 1))\n",
        "    prediction = model.predict(location_one_hot)\n",
        "    predicted_location = np.unravel_index(np.argmax(prediction), (5, 5))\n",
        "    return predicted_location\n",
        "\n",
        "# Example usage\n",
        "new_location = (0, 4)\n",
        "predicted_location = predict_bomb_location(model, new_location)\n",
        "print(f\"Predicted bomb location: {predicted_location}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Flatten\n",
        "from tensorflow.keras.utils import to_categorical\n",
        "\n",
        "# Generate dummy initial data for previous bomb locations (as a list of tuples)\n",
        "initial_bomb_locations = np.array([\n",
        "    [2, 3], [0, 1], [4, 4], [2, 2], [3, 0],\n",
        "    [1, 1], [0, 0], [4, 1], [3, 3], [1, 4]\n",
        "])\n",
        "\n",
        "# Convert the bomb locations to one-hot encoding\n",
        "def convert_to_one_hot(locations, grid_size=5):\n",
        "    one_hot = np.zeros((len(locations), grid_size, grid_size))\n",
        "    for i, (x, y) in enumerate(locations):\n",
        "        one_hot[i, x, y] = 1\n",
        "    return one_hot\n",
        "\n",
        "# Prepare the input data and labels\n",
        "X = convert_to_one_hot(initial_bomb_locations)  # Shape: (n_samples, 5, 5)\n",
        "y = convert_to_one_hot(initial_bomb_locations)  # Labels are same as inputs for simplicity\n",
        "\n",
        "# Reshape the input data to match the expected input shape for the neural network\n",
        "X = X.reshape((X.shape[0], 5, 5, 1))\n",
        "\n",
        "# Define the neural network model\n",
        "model = Sequential([\n",
        "    Flatten(input_shape=(5, 5, 1)),\n",
        "    Dense(128, activation='relu'),\n",
        "    Dense(128, activation='relu'),\n",
        "    Dense(25, activation='softmax')  # 25 units for 25 possible locations\n",
        "])\n",
        "\n",
        "# Compile the model\n",
        "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "# Initial training of the model\n",
        "model.fit(X, y.reshape((y.shape[0], 25)), epochs=50, batch_size=1)\n",
        "\n",
        "# Function to predict the bomb location\n",
        "def predict_bomb_location(model, location):\n",
        "    location_one_hot = convert_to_one_hot([location]).reshape((1, 5, 5, 1))\n",
        "    prediction = model.predict(location_one_hot)\n",
        "    predicted_location = np.unravel_index(np.argmax(prediction), (5, 5))\n",
        "    return predicted_location\n",
        "\n",
        "# Function to update the model with new data\n",
        "def update_model_with_new_location(model, new_location):\n",
        "    new_X = convert_to_one_hot([new_location]).reshape((1, 5, 5, 1))\n",
        "    new_y = convert_to_one_hot([new_location]).reshape((1, 25))\n",
        "    model.fit(new_X, new_y, epochs=1, batch_size=1)\n",
        "\n",
        "# Example usage\n",
        "new_location = (1, 2)\n",
        "predicted_location = predict_bomb_location(model, new_location)\n",
        "print(f\"Predicted bomb location: {predicted_location}\")\n",
        "\n",
        "# Update the model with the new location\n",
        "update_model_with_new_location(model, new_location)\n",
        "\n",
        "# Predict again after updating the model\n",
        "predicted_location = predict_bomb_location(model, new_location)\n",
        "print(f\"Predicted bomb location after update: {predicted_location}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QZAk2CqFh7Nx",
        "outputId": "58cde121-8ce7-43f6-9bc7-3ec35a6e1241"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/50\n",
            "10/10 [==============================] - 1s 6ms/step - loss: 3.2210 - accuracy: 0.0000e+00\n",
            "Epoch 2/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 3.0804 - accuracy: 0.4000\n",
            "Epoch 3/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.9569 - accuracy: 0.7000\n",
            "Epoch 4/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.8230 - accuracy: 0.9000\n",
            "Epoch 5/50\n",
            "10/10 [==============================] - 0s 6ms/step - loss: 2.6688 - accuracy: 1.0000\n",
            "Epoch 6/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.4840 - accuracy: 1.0000\n",
            "Epoch 7/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 2.2760 - accuracy: 1.0000\n",
            "Epoch 8/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 2.0406 - accuracy: 1.0000\n",
            "Epoch 9/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 1.7959 - accuracy: 1.0000\n",
            "Epoch 10/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 1.5379 - accuracy: 1.0000\n",
            "Epoch 11/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 1.2968 - accuracy: 1.0000\n",
            "Epoch 12/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 1.0780 - accuracy: 1.0000\n",
            "Epoch 13/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.8714 - accuracy: 1.0000\n",
            "Epoch 14/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.6808 - accuracy: 1.0000\n",
            "Epoch 15/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.5330 - accuracy: 1.0000\n",
            "Epoch 16/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.4100 - accuracy: 1.0000\n",
            "Epoch 17/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.3197 - accuracy: 1.0000\n",
            "Epoch 18/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.2548 - accuracy: 1.0000\n",
            "Epoch 19/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.2024 - accuracy: 1.0000\n",
            "Epoch 20/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.1592 - accuracy: 1.0000\n",
            "Epoch 21/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.1318 - accuracy: 1.0000\n",
            "Epoch 22/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.1093 - accuracy: 1.0000\n",
            "Epoch 23/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0927 - accuracy: 1.0000\n",
            "Epoch 24/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0799 - accuracy: 1.0000\n",
            "Epoch 25/50\n",
            "10/10 [==============================] - 0s 2ms/step - loss: 0.0701 - accuracy: 1.0000\n",
            "Epoch 26/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0611 - accuracy: 1.0000\n",
            "Epoch 27/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0549 - accuracy: 1.0000\n",
            "Epoch 28/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0486 - accuracy: 1.0000\n",
            "Epoch 29/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0440 - accuracy: 1.0000\n",
            "Epoch 30/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0396 - accuracy: 1.0000\n",
            "Epoch 31/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0360 - accuracy: 1.0000\n",
            "Epoch 32/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0330 - accuracy: 1.0000\n",
            "Epoch 33/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0303 - accuracy: 1.0000\n",
            "Epoch 34/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0279 - accuracy: 1.0000\n",
            "Epoch 35/50\n",
            "10/10 [==============================] - 0s 4ms/step - loss: 0.0257 - accuracy: 1.0000\n",
            "Epoch 36/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0239 - accuracy: 1.0000\n",
            "Epoch 37/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0224 - accuracy: 1.0000\n",
            "Epoch 38/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0209 - accuracy: 1.0000\n",
            "Epoch 39/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0195 - accuracy: 1.0000\n",
            "Epoch 40/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0183 - accuracy: 1.0000\n",
            "Epoch 41/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0172 - accuracy: 1.0000\n",
            "Epoch 42/50\n",
            "10/10 [==============================] - 0s 2ms/step - loss: 0.0162 - accuracy: 1.0000\n",
            "Epoch 43/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0152 - accuracy: 1.0000\n",
            "Epoch 44/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0144 - accuracy: 1.0000\n",
            "Epoch 45/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0136 - accuracy: 1.0000\n",
            "Epoch 46/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0130 - accuracy: 1.0000\n",
            "Epoch 47/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0123 - accuracy: 1.0000\n",
            "Epoch 48/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0117 - accuracy: 1.0000\n",
            "Epoch 49/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0112 - accuracy: 1.0000\n",
            "Epoch 50/50\n",
            "10/10 [==============================] - 0s 3ms/step - loss: 0.0106 - accuracy: 1.0000\n",
            "1/1 [==============================] - 0s 129ms/step\n",
            "Predicted bomb location: (4, 4)\n",
            "1/1 [==============================] - 0s 11ms/step - loss: 6.7673 - accuracy: 0.0000e+00\n",
            "1/1 [==============================] - 0s 24ms/step\n",
            "Predicted bomb location after update: (4, 4)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "pXghpTgpjKec"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}