import torch
import torch.nn.functional as F
import logging

def fed_avg(global_model, client_updates, frozen_layers=None):
    """
    Performs federated averaging (FedAvg) on the unfrozen layers only.
    """
    if not client_updates:
        return global_model
    
    frozen_layers = frozen_layers if frozen_layers else set()
    global_state = global_model.state_dict()

    # Average parameters for each layer not in `frozen_layers`
    for layer_name in global_state.keys():
        if layer_name not in frozen_layers:
            layer_updates = [update[layer_name].float() for update in client_updates if layer_name in update]
            if layer_updates:
                mean_update = torch.mean(torch.stack(layer_updates), dim=0)
                global_state[layer_name] = mean_update
    
    global_model.load_state_dict(global_state)
    return global_model

def evaluate_global_model(global_model, test_loader, device):
    """
    Evaluates `global_model` on the test dataset, returning accuracy.
    """
    global_model.eval()
    global_model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy

# Below are optional methods if you want to enable layer-freezing logic.

def update_stability_smooth(global_model, client_updates, stability_dict, alpha=0.98):
    """
    Updates stability metrics (how consistent parameters are across client updates).
    """
    if not client_updates:
        return stability_dict

    for name, param in global_model.named_parameters():
        # Average difference across updates for this layer
        delta_accum = torch.zeros_like(param.data)
        for update in client_updates:
            if name in update:
                delta_accum += (update[name] - param.data)
        delta_mean = delta_accum / len(client_updates)
        abs_delta = torch.abs(delta_mean)

        # If not tracked yet, initialize
        if name not in stability_dict:
            stability_dict[name] = {
                'mean': delta_mean.clone(),
                'magnitude': abs_delta.clone(),
                'stability': 1.0
            }
        else:
            stability_dict[name]['mean'] = alpha * stability_dict[name]['mean'] + (1-alpha)*delta_mean
            stability_dict[name]['magnitude'] = alpha * stability_dict[name]['magnitude'] + (1-alpha)*abs_delta
        
        # ratio => how stable is the layer
        ratio = torch.abs(stability_dict[name]['mean']) / (stability_dict[name]['magnitude'] + 1e-10)
        stability_dict[name]['stability'] = ratio.mean().item()

    return stability_dict

def apply_layer_freezing_with_hysteresis_and_cooldown(
    stability_dict, 
    frozen_layers, 
    cooldown_counters, 
    freeze_threshold=0.15, 
    unfreeze_threshold=0.25, 
    cooldown=5
):
    """
    Freezes layers if stability < freeze_threshold,
    and unfreezes them if stability > unfreeze_threshold,
    applying a cooldown to avoid oscillations.
    """
    for name, metrics in stability_dict.items():
        stability = metrics['stability']
        if name not in cooldown_counters:
            cooldown_counters[name] = 0
        else:
            cooldown_counters[name] += 1

        # Freeze if stability < freeze_threshold
        if stability < freeze_threshold and name not in frozen_layers and cooldown_counters[name] >= cooldown:
            frozen_layers.add(name)
            cooldown_counters[name] = 0
            logging.info(f"[Server] Layer {name} frozen (stability={stability:.4f}).")

        # Unfreeze if stability > unfreeze_threshold
        elif stability > unfreeze_threshold and name in frozen_layers and cooldown_counters[name] >= cooldown:
            frozen_layers.remove(name)
            cooldown_counters[name] = 0
            logging.info(f"[Server] Layer {name} unfrozen (stability={stability:.4f}).")

    return frozen_layers, cooldown_counters

def log_freezing_status(frozen_layers, stability_dict):
    """
    Logs the current freezing status of each layer.
    """
    logging.info("=== Layer Freezing Status ===")
    for name, metrics in stability_dict.items():
        status = "Frozen" if name in frozen_layers else "Active"
        stability = metrics['stability']
        logging.info(f"Layer {name}: {status}, Stability={stability:.4f}")
