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
selection_rate = 0.5
alpha = 0.1  # Dirichlet distribution parameter for non-IID
repair_threshold = 0.8*selection_rate*num_clients  # Number of active clients below which we trigger minimal repair
iid_type = "non-iid" if alpha < 1 else "iid"  # Characterize dataset
algorithm = "fedavg"
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Hyperparameters
learning_rate = 0.01
num_rounds = 100  # Reduced number of rounds for optimization
num_local_epochs = 1  # Reduced number of local epochs

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

def update_stability_smooth(global_model, client_models, stability_dict, alpha=0.98):
    """
    Updates stability metrics with smoother moving averages.
    
    Args:
        global_model (nn.Module): The global model.
        client_models (list of dict): List of client update dictionaries.
        stability_dict (dict): Dictionary to store stability metrics.
        alpha (float): Smoothing factor for moving averages.
        
    Returns:
        dict: Updated stability dictionary.
    """
    if global_model is None:
        raise ValueError("Global model is None. Stability update cannot proceed.")
    
    for name, param in global_model.named_parameters():
        delta = torch.zeros_like(param.data)
        for client_update in client_models:
            if name in client_update:
                delta += client_update[name].data - param.data
        delta /= len(client_models)
        abs_delta = torch.abs(delta)
    
        if name not in stability_dict:
            stability_dict[name] = {'mean': delta.clone(), 'magnitude': abs_delta.clone()}
        else:
            stability_dict[name]['mean'] = alpha * stability_dict[name]['mean'] + (1 - alpha) * delta
            stability_dict[name]['magnitude'] = alpha * stability_dict[name]['magnitude'] + (1 - alpha) * abs_delta
    
        ratio = torch.abs(stability_dict[name]['mean']) / (stability_dict[name]['magnitude'] + 1e-10)
        stability_dict[name]['stability'] = ratio.mean().item()
    
    return stability_dict

def categorize_layers(model):
    """
    Categorizes layers based on their type.
    
    Args:
        model (nn.Module): The model to categorize.
        
    Returns:
        dict: Mapping from layer name to layer type.
    """
    layer_types = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_types[name] = 'conv'
        elif isinstance(module, nn.BatchNorm2d):
            layer_types[name] = 'batchnorm'
        elif isinstance(module, nn.Linear):
            layer_types[name] = 'linear'
        # Add more categories as needed
    return layer_types

def apply_layer_freezing_with_hysteresis_and_cooldown(stability_dict, frozen_layers, cooldown_counters, 
                                                      freeze_threshold=0.7, unfreeze_threshold=0.8, 
                                                      cooldown=3):
    """
    Freezes or unfrezes layers based on stability thresholds with hysteresis and cooldown to prevent oscillations.
    
    Args:
        stability_dict (dict): Dictionary containing stability metrics for each layer.
        frozen_layers (set): Set of currently frozen layer names.
        cooldown_counters (dict): Dictionary tracking cooldown for each layer.
        freeze_threshold (float): Threshold below which layers are frozen.
        unfreeze_threshold (float): Threshold above which layers are unfrozen.
        cooldown (int): Number of rounds to wait before changing layer state.
        
    Returns:
        set: Updated set of frozen layer names.
        dict: Updated cooldown counters.
    """
    for name, metrics in stability_dict.items():
        stability = metrics.get('stability', 1.0)  # Default to stable if not present
        
        if name not in cooldown_counters:
            cooldown_counters[name] = 0
        else:
            cooldown_counters[name] += 1
        
        if stability < freeze_threshold and name not in frozen_layers and cooldown_counters[name] >= cooldown:
            frozen_layers.add(name)
            cooldown_counters[name] = 0  # Reset cooldown
            logging.info(f"Layer {name} has been frozen (stability: {stability:.4f}).")
        
        elif stability > unfreeze_threshold and name in frozen_layers and cooldown_counters[name] >= cooldown:
            frozen_layers.remove(name)
            cooldown_counters[name] = 0  # Reset cooldown
            logging.info(f"Layer {name} has been unfrozen (stability: {stability:.4f}).")
        
        # If not meeting conditions, do not change state and cooldown continues
    return frozen_layers, cooldown_counters

def log_freezing_status(frozen_layers, stability_dict):
    """
    Logs the current freezing status of all layers.
    
    Args:
        frozen_layers (set): Set of currently frozen layer names.
        stability_dict (dict): Dictionary containing stability metrics for each layer.
    """
    logging.info("Current Freezing Status:")
    for name, metrics in stability_dict.items():
        status = "Frozen" if name in frozen_layers else "Active"
        stability = metrics.get('stability', 1.0)
        logging.info(f"Layer {name}: {status}, Stability: {stability:.4f}")

def get_client_update(local_model, frozen_layers):
    return {name: param for name, param in local_model.state_dict().items() if name not in frozen_layers}

def get_transmitted_data_size(client_update, frozen_layers):
    """
    Calculates the size of the transmitted data based on the client update dictionary.

    Args:
        client_update (dict): A dictionary of parameter tensors from the client.
        frozen_layers (set): A set of layer names that are frozen and should not be included.

    Returns:
        int: Total size of the transmitted data in bytes.
    """
    
    if not isinstance(client_update, dict):
        raise TypeError(f"Expected client_update to be a dict, but got {type(client_update)}")
    
    transmitted_size = sum(
        param.numel() * param.element_size() for name, param in client_update.items() if name not in frozen_layers
    )
    return transmitted_size

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

# Initialize cooldown counters
cooldown_counters = {}

# Categorize layers for prioritization (if implementing prioritized freezing)
layer_types = categorize_layers(global_model)
#frozen_layers.add("layer1.0.conv1.weight")  # or any param name from model.state_dict()

for round in range(num_rounds):
    logging.info(f"Starting round {round + 1}")
    start_time = time.time()

    round_loss = []
    round_accuracy = []
    client_updates = []

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

        # Extract updates for unfrozen layers
        client_update = get_client_update(local_model, frozen_layers)
        client_update_size = sum(param.numel() * param.element_size() for param in client_update.values())
        logging.info(f"Client {client_id + 1}: Transmitted {client_update_size / (1024 * 1024):.2f} MB")

        # Append client updates
        client_updates.append(client_update)

    # Step 2: Aggregate models
    if client_updates:
        global_model = fed_avg_with_frozen_layers(global_model, client_updates, frozen_layers)
    else:
        logging.warning("No client models received. Skipping aggregation.")
        continue  # Skip to the next round

    # Step 3: Update stability with smoothing
    if not client_updates:
        logging.warning("No client models available for stability update.")
    else:
        stability_dict = update_stability_smooth(global_model, client_updates, stability_dict, alpha=0.98)

    # Apply layer freezing with hysteresis and cooldown
    frozen_layers, cooldown_counters = apply_layer_freezing_with_hysteresis_and_cooldown(
        stability_dict, frozen_layers, cooldown_counters, 
        freeze_threshold=0.15, 
        unfreeze_threshold=0.25, 
        cooldown=5
    )
    logging.info(f"Round {round+1} - Currently frozen layers: {frozen_layers}")
    print(f"Round {round+1} - Currently frozen layers: {frozen_layers}")

    # Log current freezing status
    log_freezing_status(frozen_layers, stability_dict)

    # Step 4: Evaluate model
    test_accuracy = evaluate_model(test_loader, global_model, device)
    logging.info(f"Test Accuracy after round {round + 1}: {test_accuracy:.2f}%")

    elapsed_time = time.time() - start_time
    logging.info(f"Round {round + 1} completed in {elapsed_time:.2f} seconds\n")

    # Calculate total data transmitted in this round
    total_round_data = sum(get_transmitted_data_size(client_update, frozen_layers) for client_update in client_updates)
    logging.info(f"Total data transmitted in round {round + 1}: {total_round_data / (1024 * 1024):.2f} MB")


logging.info("Federated learning process finished successfully.")  
