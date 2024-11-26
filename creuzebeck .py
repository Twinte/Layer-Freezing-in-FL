import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, Subset
import time
import logging
from collections import Counter
import math
from datetime import datetime
import random

# Characteristics for logging
num_clients = 30
selection_rate = 1
alpha = 0.1  # Dirichlet distribution parameter for non-IID
repair_threshold = 0.8*selection_rate*num_clients  # Number of active clients below which we trigger minimal repair
iid_type = "non-iid" if alpha < 1 else "iid"  # Characterize dataset
algorithm = "fedavg"
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Log file name format: includes date, algorithm, non-IID/IID type, and number of clients
log_file_name = f"federated_learning_{algorithm}_{iid_type}_{num_clients}clients_{date_str}.txt"

# Ensure the directory for logs exists
log_directory = './experiment_logs'  # You can change this path as needed
os.makedirs(log_directory, exist_ok=True)  # Create directory if it doesn't exist
log_file_path = os.path.join(log_directory, log_file_name)

# Set up logging with dynamic file name using full log_file_path
logging.basicConfig(filename=log_file_path, level=logging.INFO)

# Client failure parameters
failure_probability = 0.00  # 5% chance of failure per round
rejoin_probability = 0.3   # 30% chance that a failed client will rejoin in the next round

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformations for the dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Download and load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader for the test dataset
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

def get_model_size(model):
    """Calculates the size of the model in bytes."""
    total_size = 0
    for param in model.parameters():
        total_size += param.element_size() * param.nelement()  # Size of each element * number of elements
    return total_size

# Function to split dataset using Dirichlet distribution for non-IID split
def split_dataset_by_dirichlet(dataset, num_clients, alpha):
    indices = np.arange(len(dataset))
    label_array = np.array([dataset[i][1] for i in indices])  # Get labels from the dataset
    client_indices = [[] for _ in range(num_clients)]

    # Split data for each class based on Dirichlet distribution
    for label in np.unique(label_array):
        class_indices = indices[label_array == label]
        np.random.shuffle(class_indices)
        class_split = np.random.dirichlet([alpha] * num_clients) * len(class_indices)
        class_split = np.round(class_split).astype(int)
        class_split = np.cumsum(class_split).astype(int)

        start = 0
        for client_id in range(num_clients):
            client_indices[client_id].extend(class_indices[start:class_split[client_id]])
            start = class_split[client_id]

    # Return indices for each client
    return client_indices

# Function to calculate entropy given label counts
def calculate_entropy(label_counts):
    total_samples = sum(label_counts.values())
    entropy = 0.0
    for count in label_counts.values():
        if count > 0:
            p_i = count / total_samples
            entropy -= p_i * math.log2(p_i)
    return entropy

# Non-IID data partition
client_data_indices = split_dataset_by_dirichlet(train_dataset, num_clients, alpha)

# Create a DataLoader for each client
client_loaders = [DataLoader(Subset(train_dataset, indices), batch_size=64, shuffle=True) 
                  for indices in client_data_indices]

# Calculate, log, and store entropy for each client
client_entropies = []
client_sample_sizes = []  # Store the dataset sizes for each client

for i, indices in enumerate(client_data_indices):
    # Extract labels for the current client
    labels = [train_dataset[idx][1] for idx in indices]
    # Count occurrences of each label
    label_counts = Counter(labels)
    # Calculate entropy
    entropy = calculate_entropy(label_counts)
    client_entropies.append((i, entropy))  # Store (client_id, entropy) tuples

    # Get the sample size (number of samples in each client's dataset)
    client_sample_size = len(client_loaders[i].dataset)
    client_sample_sizes.append(client_sample_size)  # Store client sample size

    # Log the entropy and sample size
    logging.info(f"Client {i+1} Label Entropy: {entropy:.4f}, Sample Size: {client_sample_size}")
    print(f"Client {i+1} Label Entropy: {entropy:.4f}, Sample Size: {client_sample_size}")

# Step 1: Normalize entropy and dataset size values

# Normalize entropy values
entropy_values = np.array([entropy for _, entropy in client_entropies])
min_entropy = np.min(entropy_values)
max_entropy = np.max(entropy_values)
normalized_entropies = (entropy_values - min_entropy) / (max_entropy - min_entropy + 1e-10)  # Avoid division by zero

# Normalize dataset sizes
total_samples = sum(client_sample_sizes)
normalized_sample_sizes = np.array(client_sample_sizes) / total_samples  # Normalize based on total number of samples

# Step 2: Combine entropy and dataset size into a weighted score
weight_entropy = 0.5  # You can adjust the weight for entropy
weight_sample_size = 0.5  # You can adjust the weight for sample size

combined_scores = (weight_entropy * normalized_entropies) + (weight_sample_size * normalized_sample_sizes)

# Step 3: Rank clients based on the combined score in descending order
client_scores = [(client_id, score) for client_id, score in zip(range(len(client_entropies)), combined_scores)]
client_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by combined score

# Step 4: Select the top 20% of clients based on the combined score
num_selected_clients = max(1, int(selection_rate * len(client_scores)))  # Ensure at least one client is selected
selected_clients = [client_id for client_id, _ in client_scores[:num_selected_clients]]

# Log and print the selected clients
logging.info(f"Selected top {num_selected_clients} clients for training based on combined entropy and dataset size.")
print(f"Selected top {num_selected_clients} clients for training: {selected_clients}")

# Debug: Print selected clients with their combined scores
for client_id, score in client_scores[:num_selected_clients]:
    logging.info(f"Selected Client {client_id + 1}: Combined Score {score:.4f}")
    print(f"Selected Client {client_id + 1}: Combined Score {score:.4f}")

# Define the CNN architecture for CIFAR-10 (reduced complexity for optimization)
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CIFAR10_ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFAR10_ResNet, self).__init__()
        self.layer1 = self._make_layer(3, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Hyperparameters
learning_rate = 0.001
num_rounds = 100  # Reduced number of rounds for optimization
num_local_epochs = 5  # Reduced number of local epochs

# FedAvg Algorithm to average the model weights modified for freezing layers
def fed_avg_with_frozen_layers(global_model, client_updates, frozen_layers):
    """
    Aggregates only the unfrozen layers from client updates and updates the global model.
    """
    if not client_updates:
        logging.warning("No client updates received. Global model remains unchanged.")
        return global_model  # Return the current global model if no updates are provided.

    global_state = global_model.state_dict()
    for name in global_state.keys():
        if name not in frozen_layers:  # Only aggregate unfrozen layers
            updates = [client[name].float() for client in client_updates if name in client]
            if updates:  # Ensure there are updates for the layer
                global_state[name] = torch.mean(torch.stack(updates), dim=0)
    global_model.load_state_dict(global_state)
    return global_model

def update_stability(global_model, client_models, stability_dict, alpha=0.95):
    if global_model is None:
        raise ValueError("Global model is None. Stability update cannot proceed.")
    
    for name, param in global_model.named_parameters():
        delta = torch.zeros_like(param.data)
        for client_model in client_models:
            delta += client_model[name] - param.data

        delta /= len(client_models)
        abs_delta = torch.abs(delta)

        # Update moving averages for mean and magnitude of weight updates
        if name not in stability_dict:
            stability_dict[name] = {'mean': delta, 'magnitude': abs_delta}
        else:
            stability_dict[name]['mean'] = alpha * stability_dict[name]['mean'] + (1 - alpha) * delta
            stability_dict[name]['magnitude'] = alpha * stability_dict[name]['magnitude'] + (1 - alpha) * abs_delta

        # Calculate stability index
        ratio = torch.abs(stability_dict[name]['mean']) / (stability_dict[name]['magnitude'] + 1e-10)
        stability_dict[name]['stability'] = ratio.mean().item()

    return stability_dict

def apply_layer_freezing(stability_dict, frozen_layers, stability_threshold=0.1):
    for name, metrics in stability_dict.items():
        if metrics['stability'] < stability_threshold and name not in frozen_layers:
            frozen_layers.add(name)  # Freeze this layer
    return frozen_layers

def get_client_update(local_model, frozen_layers):
    return {name: param for name, param in local_model.state_dict().items() if name not in frozen_layers}

def get_transmitted_data_size(model, frozen_layers):
    transmitted_size = sum(
        p.numel() * p.element_size() for name, p in model.named_parameters() if name not in frozen_layers
    )
    return transmitted_size  # In bytes

def train_client_with_freezing(client_loader, model, criterion, optimizer, frozen_layers, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in client_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        
        # Skip gradients for frozen layers
        for name, param in model.named_parameters():
            if name in frozen_layers:
                param.requires_grad = False

        loss = criterion(outputs, labels)
        loss.backward()
        
        # Unfreeze layers temporarily for optimizer step
        for name, param in model.named_parameters():
            if name in frozen_layers:
                param.requires_grad = True
        
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(client_loader), accuracy


# Evaluate the model
def evaluate_model(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100. * correct / total
    return accuracy

# Track active clients (start with all clients active)
client_active_status = [True] * num_clients  # Initially, all clients are active

# Main federated learning loop
global_model = CIFAR10_ResNet().to(device)
criterion = nn.CrossEntropyLoss()

# Initialize log
logging.info(f"Starting federated learning with {num_clients} clients")

stability_dict = {}
frozen_layers = set()

for round in range(num_rounds):
    logging.info(f"Starting round {round + 1}")
    start_time = time.time()

    round_loss = []
    round_accuracy = []
    client_models = []

    # Step 1: Train active clients
    for client_id in selected_clients:
        local_model = CIFAR10_ResNet().to(device)
        local_model.load_state_dict(global_model.state_dict())
        optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

        # Train client with layer freezing
        client_loss, client_accuracy = train_client_with_freezing(
            client_loaders[client_id], local_model, criterion, optimizer, frozen_layers, device
        )
        round_loss.append(client_loss)
        round_accuracy.append(client_accuracy)

        #Extract upates for unfrozen clients
        client_update = get_client_update(local_model, frozen_layers)
        client_update_size = sum(param.numel() * param.element_size() for param in client_update.values())
        logging.info(f"Client {client_id}: Transmitted {client_update_size / (1024 * 1024):.2f} MB")

        #Append client updates(not full models)
        client_models.append(client_update)

    
    # Step 2: Aggregate models
    # Aggregate models with frozen layers
    if client_models:
        global_model = fed_avg_with_frozen_layers(global_model, client_models, frozen_layers)
    else:
        logging.warning("No client models received. Skipping aggregation.")
        continue  # Skip to the next round

        #global_model_size = get_model_size(global_model)
        #logging.info(f"Global model size after this round: {global_model_size / (1024 * 1024):.2f} MB")
    
    # Step 3: Update stability and apply freezing
    if not client_models:
        logging.warning("No client models available for stability update.")
    else:
        stability_dict = update_stability(global_model, client_models, stability_dict)
    frozen_layers = apply_layer_freezing(stability_dict, frozen_layers)
    
    # Step 4: Evaluate model
    test_accuracy = evaluate_model(test_loader, global_model, device)
    logging.info(f"Test Accuracy after round {round + 1}: {test_accuracy:.2f}%")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Round {round + 1} completed in {elapsed_time:.2f} seconds\n")
    total_round_data = sum(get_transmitted_data_size(client_model, frozen_layers) for client_model in client_models)
    logging.info(f"Total data transmitted in round {round + 1}: {total_round_data / (1024 * 1024):.2f} MB")


logging.info("Federated learning process finished successfully.")
