import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def load_cifar10_dataset(data_dir='./data'):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def split_dataset_by_dirichlet(dataset, num_clients, alpha):
    indices = np.arange(len(dataset))
    labels = np.array([dataset[i][1] for i in indices])
    client_indices = [[] for _ in range(num_clients)]

    for label in np.unique(labels):
        label_indices = indices[labels == label]
        np.random.shuffle(label_indices)

        # Partition this class using Dirichlet
        class_split = np.random.dirichlet([alpha] * num_clients) * len(label_indices)
        class_split = np.round(class_split).astype(int)
        class_split = np.cumsum(class_split).astype(int)

        start = 0
        for client_id in range(num_clients):
            client_indices[client_id].extend(label_indices[start:class_split[client_id]])
            start = class_split[client_id]
    
    return client_indices

def create_client_loaders(train_dataset, client_data_indices, batch_size=64):
    client_loaders = []
    for idxs in client_data_indices:
        subset = Subset(train_dataset, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    return client_loaders

def filter_dataset_by_classes(dataset, keep_class_list):
    """
    Returns a new Subset of `dataset` containing only samples 
    whose labels are in `keep_class_list`.
    """
    keep_class_list = set(keep_class_list)  # for quick membership checking
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in keep_class_list:
            indices.append(i)
    return Subset(dataset, indices)
