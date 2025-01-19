import torch

def train_local_model(model, dataloader, criterion, optimizer, device, num_local_epochs=1, frozen_layers=None):
    """
    Trains `model` locally on the client's `dataloader` for `num_local_epochs` epochs.
    If `frozen_layers` is not None, those layers do NOT update (their gradients are zero).
    
    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    model.to(device)
    total_samples = 0
    correct = 0
    running_loss = 0.0

    # If no freezing is specified, treat it as an empty set
    frozen_layers = frozen_layers if frozen_layers else set()

    for epoch in range(num_local_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Freeze specified layers (disable gradient)
            for name, param in model.named_parameters():
                if name in frozen_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    accuracy = 100.0 * correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy

def get_client_update(local_model, frozen_layers=None):
    """
    Returns a dictionary containing the parameters (state_dict) of layers 
    that are not in `frozen_layers`.
    """
    update_dict = {}
    for name, param in local_model.state_dict().items():
        if not frozen_layers or name not in frozen_layers:
            update_dict[name] = param.cpu().clone()
    return update_dict

def get_transmitted_data_size(param_dict):
    """
    Computes size in bytes of all tensors in `param_dict`.
    """
    total_bytes = 0
    for tensor in param_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes
