# main.py
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import json
import matplotlib.pyplot as plt

def load_hyperparameters(param_file):
    try:
        with open(param_file, 'r') as file:
            hyperparameters = json.load(file)
        return hyperparameters
    except FileNotFoundError:
        print(f"Error: Hyperparameter file '{param_file}' not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in '{param_file}'. Please check the file format.")
        raise


def generate_dataset(seed, train_size, test_size):
    # Set random seed for reproducibility
    np.random.seed(seed)

    def int_to_binary_list(num, num_bits):
        binary_str = bin(num)[2:].zfill(num_bits)
        return [int(bit) for bit in binary_str]

    # Generate training data
    train_dataset = []
    for _ in range(train_size):
        A = np.random.randint(0, 2**8)  # Random 8-bit integer
        B = np.random.randint(0, 2**8)  # Random 8-bit integer
        C = A * B
        A_bits = int_to_binary_list(A, 8)
        B_bits = int_to_binary_list(B, 8)
        C_bits = int_to_binary_list(C, 16)
        input_sequence = A_bits + B_bits + [0]  # Concatenate A, B, and a zero
        target_sequence = [0] + C_bits  # Add a zero at the beginning for junk
        train_dataset.append((input_sequence, target_sequence))

    # Generate test data
    test_dataset = []
    for _ in range(test_size):
        A = np.random.randint(0, 2**8)  # Random 8-bit integer
        B = np.random.randint(0, 2**8)  # Random 8-bit integer
        C = A * B
        A_bits = int_to_binary_list(A, 8)
        B_bits = int_to_binary_list(B, 8)
        C_bits = int_to_binary_list(C, 16)
        input_sequence = B_bits + A_bits + [0]  # Swap A and B, concatenate, and add a zero
        target_sequence = [0] + C_bits  # Add a zero at the beginning for junk
        test_dataset.append((input_sequence, target_sequence))

    return train_dataset, test_dataset

def train_rnn(train_dataset, test_dataset, hyperparameters):
    # Unpack hyperparameters
    input_size = hyperparameters["input_size"]
    output_size = hyperparameters["output_size"]
    hidden_size = hyperparameters["hidden_size"]
    learning_rate = hyperparameters["learning_rate"]
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]

    # Extract input and target sequences from the training dataset
    input_sequences, target_sequences = zip(*train_dataset)

    # Convert sequences to numpy arrays
    X_train = np.array(input_sequences)
    y_train = np.array(target_sequences)

    # Extract input and target sequences from the test dataset
    input_sequences_test, target_sequences_test = zip(*test_dataset)

    # Convert sequences to numpy arrays
    X_test = np.array(input_sequences_test)
    y_test = np.array(target_sequences_test)

    # Build the RNN model
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(input_size, 1)))
    model.add(Dense(output_size, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # Print model summary
    model.summary()

    # Lists to store training and test losses for plotting
    train_losses = []
    test_losses = []
    test_losses_swapped = []

    # Train the model
    for epoch in range(epochs):
        # Train the model for one epoch
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)

        # Evaluate the model on the training set
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss}")
        train_losses.append(train_loss)

        # Evaluate the model on the test set
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss: {test_loss}")
        test_losses.append(test_loss)

        # Swap A and B for the test set
        X_test_swapped = np.array([sequence[:8] + sequence[16:] + sequence[8:16] for sequence in X_test])
        test_loss_swapped = model.evaluate(X_test_swapped, y_test, verbose=0)
        test_losses_swapped.append(test_loss_swapped)
        print(f"Epoch {epoch + 1}/{epochs} - Test Loss (Swapped): {test_loss_swapped}")

    # Plot training and test losses
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.plot(range(1, epochs + 1), test_losses_swapped, label='Test Loss Swapped')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Save the figure
    plt.savefig('loss_plot.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Trains an RNN to perform multiplication of binary integers A * B = C")
    parser.add_argument("--param", default="param/param.json", help="file containing hyperparameters")
    parser.add_argument("--train-size", type=int, help="size of the generated training set")
    parser.add_argument("--test-size", type=int, help="size of the generated test set")
    parser.add_argument("--seed", type=int, help="random seed used for creating the datasets")
    args = parser.parse_args()

    # Load hyperparameters from the specified file
    hyperparameters = load_hyperparameters(args.param)

    # Generate the dataset
    train_dataset, test_dataset = generate_dataset(args.seed, args.train_size, args.test_size)

    # Split the test dataset
    split_index = int(0.8 * len(test_dataset))
    train_dataset += test_dataset[:split_index]
    test_dataset = test_dataset[split_index:]

    # Train the RNN
    train_rnn(train_dataset, test_dataset, hyperparameters)


if __name__ == "__main__":
    main()
