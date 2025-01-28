import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# GRU Model Definition
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)  # Predict at each timestamp
        return out

# Data Loading and Preprocessing
def load_and_pad_data(file_paths, device):
    data = []
    labels = []
    
    for file in file_paths:
        df = pd.read_csv(file)

        # Process Real and Imaginary features
        real_values = np.array([np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else np.array([]) for x in df["Real"]])
        imag_values = np.array([np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else np.array([]) for x in df["Imaginary"]])
        
        assert all(len(r) == len(real_values[0]) for r in real_values), "Real feature lengths inconsistent."
        assert all(len(i) == len(imag_values[0]) for i in imag_values), "Imaginary feature lengths inconsistent."
        
        combined_features = np.hstack((real_values, imag_values, df[["Direction", "Voice Detected", "Decibels"]].values))
        label = df["label"].values
        
        # Check for out-of-bounds values and replace with 0
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
        label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)

        data.append(torch.tensor(combined_features, dtype=torch.float32))
        labels.append(torch.tensor(label, dtype=torch.long))

    # Pad sequences to the max length across all files
    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return padded_data.to(device), padded_labels.to(device)

# Training Function
def train(model, criterion, optimizer, train_data, train_labels, val_data, val_labels, num_epochs):
    train_losses = []
    val_losses = []
    clip_value = 5  # Gradient clipping threshold

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_data)
        
        # Mask padding values
        train_mask = train_labels != 0
        masked_outputs = outputs[train_mask]
        masked_labels = train_labels[train_mask]

        # Compute loss
        loss = criterion(masked_outputs, masked_labels)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        train_losses.append(loss.item())

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_mask = val_labels != 0
            masked_val_outputs = val_outputs[val_mask]
            masked_val_labels = val_labels[val_mask]
            val_loss = criterion(masked_val_outputs, masked_val_labels)
            val_losses.append(val_loss.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    # Plot training and validation loss
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

def evaluate(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        predictions = torch.argmax(outputs, dim=-1)

    # Flatten predictions and true labels
    predictions = predictions.view(-1).cpu().numpy()
    true_labels = test_labels.view(-1).cpu().numpy()

    # Mask padding values (assume 0 is padding label)
    mask = true_labels != 0
    predictions = predictions[mask]
    true_labels = true_labels[mask]

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Main function remains mostly unchanged
# (loading data, splitting into train/test/val, model initialization, etc.)

# Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    data_dir = "all_radar_trials"
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    features, labels = load_and_pad_data(file_paths, device)

    # Split data into train, validation, and test sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

    # Model Parameters
    input_size = features.size(-1)
    hidden_size = 64
    output_size = 2
    num_layers = 2
    dropout = 0.2
    num_epochs = 10
    learning_rate = 0.001

    model = SimpleGRU(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, criterion, optimizer, train_data, train_labels, val_data, val_labels, num_epochs)

    # Evaluate the model
    evaluate(model, test_data, test_labels)

if __name__ == "__main__":
    main()
