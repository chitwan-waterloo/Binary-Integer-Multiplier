import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

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
    np.random.seed(seed)

    def int_to_binary_list(num, num_bits):
        binary_str = bin(num)[2:].zfill(num_bits)
        return [int(bit) for bit in binary_str]

    train_dataset = []
    for _ in range(train_size):
        A = np.random.randint(0, 2**8)
        B = np.random.randint(0, 2**8)
        C = A * B
        A_bits = int_to_binary_list(A, 8)
        B_bits = int_to_binary_list(B, 8)
        C_bits = int_to_binary_list(C, 16)
        input_sequence = A_bits + B_bits + [0]
        target_sequence = [0] + C_bits
        train_dataset.append((input_sequence, target_sequence))

    test_dataset = []
    for _ in range(test_size):
        A = np.random.randint(0, 2**8)
        B = np.random.randint(0, 2**8)
        C = A * B
        A_bits = int_to_binary_list(A, 8)
        B_bits = int_to_binary_list(B, 8)
        C_bits = int_to_binary_list(C, 16)
        input_sequence = B_bits + A_bits + [0]
        target_sequence = [0] + C_bits
        test_dataset.append((input_sequence, target_sequence))

    return train_dataset, test_dataset

def train_rnn(train_dataset, test_dataset, hyperparameters):
    input_size = hyperparameters["input_size"]
    hidden_size = hyperparameters["hidden_size"]
    output_size = hyperparameters["output_size"]
    learning_rate = hyperparameters["learning_rate"]
    epochs = hyperparameters["epochs"]
    batch_size = hyperparameters["batch_size"]

    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(train_dataset), batch_size):
            inputs, targets = zip(*train_dataset[i:i+batch_size])
            inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(2)
            targets = torch.tensor(targets, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        inputs_test, targets_test = zip(*test_dataset)
        inputs_test = torch.tensor(inputs_test, dtype=torch.float32).unsqueeze(2)
        targets_test = torch.tensor(targets_test, dtype=torch.long)
        outputs_test = model(inputs_test)
        test_loss = criterion(outputs_test, targets_test.view(-1))
        print(f"Test Loss: {test_loss.item()}")

        # Swap A and B for the test set
        inputs_test_swapped = torch.tensor([sequence[:8] + sequence[16:] + sequence[8:16] for sequence in inputs_test], dtype=torch.float32).unsqueeze(2)
        outputs_test_swapped = model(inputs_test_swapped)
        test_loss_swapped = criterion(outputs_test_swapped, targets_test.view(-1))
        print(f"Test Loss (Swapped): {test_loss_swapped.item()}")

def main():
    parser = argparse.ArgumentParser(description="Trains an RNN to perform multiplication of binary integers A * B = C")
    parser.add_argument("--param", default="param/param.json", help="file containing hyperparameters")
    parser.add_argument("--train-size", type=int, help="size of the generated training set")
    parser.add_argument("--test-size", type=int, help="size of the generated test set")
    parser.add_argument("--seed", type=int, help="random seed used for creating the datasets")
    args = parser.parse_args()

    hyperparameters = load_hyperparameters(args.param)
    train_dataset, test_dataset = generate_dataset(args.seed, args.train_size, args.test_size)
    
    # Split the test dataset
    split_index = int(0.8 * len(test_dataset))
    train_dataset += test_dataset[:split_index]
    test_dataset = test_dataset[split_index:]

    train_rnn(train_dataset, test_dataset, hyperparameters)

if __name__ == "__main__":
    main()