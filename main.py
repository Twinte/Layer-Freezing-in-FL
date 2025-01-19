import argparse
import logging
import os
import random
import time
import numpy as np
import torch
from datetime import datetime
from collections import Counter

from data_partition import (
    load_cifar10_dataset, 
    split_dataset_by_dirichlet, 
    create_client_loaders,
    filter_dataset_by_classes
)
from model import CIFAR10_ResNet
from client import train_local_model, get_client_update, get_transmitted_data_size
from server import (
    fed_avg,
    evaluate_global_model,
    update_stability_smooth,
    apply_layer_freezing_with_hysteresis_and_cooldown,
    log_freezing_status
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_entropy(label_counts):
    total = sum(label_counts.values())
    entropy = 0.0
    for count in label_counts.values():
        if count > 0:
            p_i = count / total
            entropy -= p_i * np.log2(p_i)
    return entropy

def parse_class_list(class_str):
    """
    Parses a comma-separated string of class indices (e.g., "0,1,2,3,4") 
    into a list of integers.
    """
    if not class_str:
        return []
    return [int(x) for x in class_str.split(',')]

def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Concept Drift Example")
    # Basic FL params
    parser.add_argument('--num_clients', type=int, default=30)
    parser.add_argument('--selection_rate', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--num_local_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./experiment_logs')

    # Layer freezing toggles
    parser.add_argument('--enable_freezing', action='store_true', default=False,
                        help='Enable stability-based layer freezing')
    parser.add_argument('--freeze_threshold', type=float, default=0.15)
    parser.add_argument('--unfreeze_threshold', type=float, default=0.25)
    parser.add_argument('--cooldown', type=int, default=5)

    # Concept drift toggles
    parser.add_argument('--enable_concept_drift', action='store_true', default=False,
                        help='Enable concept drift simulation (switching between scenario A and B).')
    parser.add_argument('--scenario_a_classes', type=str, default="0,1,2,3,4",
                        help='Comma-separated class indices for Scenario A.')
    parser.add_argument('--scenario_b_classes', type=str, default="5,6,7,8,9",
                        help='Comma-separated class indices for Scenario B.')
    parser.add_argument('--concept_drift_switches', type=int, default=1,
                        help='Number of times to switch between Scenario A and Scenario B.')
    parser.add_argument('--concept_drift_rounds_per_scenario', type=int, default=3,
                        help='Number of consecutive rounds to train in current scenario before switching.')

    args = parser.parse_args()
    set_seed(args.seed)

    # Logging setup
    os.makedirs(args.log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    iid_type = "non-iid" if args.alpha < 1 else "iid"
    log_file = f"fedlearning_{iid_type}_{args.num_clients}clients_{date_str}.log"
    log_path = os.path.join(args.log_dir, log_file)
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info("Starting Federated Learning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1) Load full CIFAR-10 dataset
    full_train_dataset, full_test_dataset = load_cifar10_dataset(args.data_dir)

    # --- CONCEPT DRIFT SETUP ---
    if args.enable_concept_drift:
        # Parse class lists for scenario A and scenario B
        scenario_a_list = parse_class_list(args.scenario_a_classes)
        scenario_b_list = parse_class_list(args.scenario_b_classes)

        # Filter the original training sets to create scenario-specific datasets
        scenario_a_dataset = filter_dataset_by_classes(full_train_dataset, scenario_a_list)
        scenario_b_dataset = filter_dataset_by_classes(full_train_dataset, scenario_b_list)

        # Partition each scenario dataset among the clients
        scenario_a_indices = split_dataset_by_dirichlet(scenario_a_dataset, args.num_clients, args.alpha)
        scenario_b_indices = split_dataset_by_dirichlet(scenario_b_dataset, args.num_clients, args.alpha)

        scenario_a_loaders = create_client_loaders(scenario_a_dataset, scenario_a_indices, args.batch_size)
        scenario_b_loaders = create_client_loaders(scenario_b_dataset, scenario_b_indices, args.batch_size)

        # Also create scenario-specific test sets to measure local performance
        scenario_a_test_set = filter_dataset_by_classes(full_test_dataset, scenario_a_list)
        scenario_b_test_set = filter_dataset_by_classes(full_test_dataset, scenario_b_list)
        scenario_a_test_loader = torch.utils.data.DataLoader(scenario_a_test_set, batch_size=args.batch_size, shuffle=False)
        scenario_b_test_loader = torch.utils.data.DataLoader(scenario_b_test_set, batch_size=args.batch_size, shuffle=False)

        # We'll start in Scenario A (arbitrary)
        current_scenario = 'A'
        scenario_switches_remaining = args.concept_drift_switches
        rounds_in_current_scenario = 0
        
    else:
        # No concept drift => use full dataset
        scenario_a_dataset = full_train_dataset
        scenario_a_indices = split_dataset_by_dirichlet(scenario_a_dataset, args.num_clients, args.alpha)
        scenario_a_loaders = create_client_loaders(scenario_a_dataset, scenario_a_indices, args.batch_size)

        # For testing, we'll use the full test set
        test_loader_full = torch.utils.data.DataLoader(full_test_dataset, batch_size=args.batch_size, shuffle=False)

        current_scenario = 'A'
        scenario_switches_remaining = 0
        rounds_in_current_scenario = 0

    # Helper for picking the active scenario’s loaders
    def get_active_client_loaders():
        if not args.enable_concept_drift:
            return scenario_a_loaders  # only one scenario in this case
        return scenario_a_loaders if current_scenario == 'A' else scenario_b_loaders

    # Helper for picking the active scenario’s test loader (when concept drift is ON)
    def get_active_test_loader():
        if not args.enable_concept_drift:
            # If concept drift is disabled, we test on the FULL dataset 
            return test_loader_full  
        if current_scenario == 'A':
            return scenario_a_test_loader
        else:
            return scenario_b_test_loader

    def select_clients_for_scenario(active_loaders, active_dataset):
        from collections import Counter
        client_entropies = []
        client_sizes = []
        for loader in active_loaders:
            subset_indices = loader.dataset.indices  # Subset
            labels = [active_dataset[i][1] for i in subset_indices]
            count_dict = Counter(labels)
            ent = calculate_entropy(count_dict)
            client_entropies.append(ent)
            client_sizes.append(len(subset_indices))

        # Normalize
        ent_arr = np.array(client_entropies)
        min_ent, max_ent = ent_arr.min(), ent_arr.max()
        denom_ent = max_ent - min_ent if (max_ent - min_ent) > 0 else 1e-10
        norm_ent = (ent_arr - min_ent) / denom_ent

        size_arr = np.array(client_sizes)
        total_size = size_arr.sum()
        norm_sizes = size_arr / (total_size + 1e-10)

        # Combine
        combined = 0.5 * norm_ent + 0.5 * norm_sizes
        sorted_indices = np.argsort(-combined)
        num_selected = max(1, int(args.selection_rate * args.num_clients))
        selected = sorted_indices[:num_selected].tolist()
        return selected

    # Initial selection for scenario
    if args.enable_concept_drift and current_scenario == 'B':
        active_loaders = scenario_b_loaders
        selected_clients = select_clients_for_scenario(scenario_b_loaders, scenario_b_dataset)
    else:
        active_loaders = scenario_a_loaders
        selected_clients = select_clients_for_scenario(scenario_a_loaders, scenario_a_dataset)

    logging.info(f"Initial scenario = {current_scenario}, selected clients => {selected_clients}")

    # 3) Initialize global model
    global_model = CIFAR10_ResNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # 4) If freezing is enabled, track stability info
    stability_dict = {}
    frozen_layers = set()
    cooldown_counters = {}

    # If concept drift is disabled, we already have `test_loader_full` created above.
    # Otherwise, we use scenario-based test loaders in get_active_test_loader().

    # 5) Main FL loop
    for round_idx in range(args.num_rounds):
        logging.info(f"=== Global Round {round_idx + 1} (Scenario {current_scenario}) ===")
        start_time = time.time()

        # Possibly switch scenario if concept drift is enabled
        if args.enable_concept_drift:
            if rounds_in_current_scenario >= args.concept_drift_rounds_per_scenario and scenario_switches_remaining > 0:
                # Switch from A->B or B->A
                current_scenario = 'B' if current_scenario == 'A' else 'A'
                scenario_switches_remaining -= 1
                rounds_in_current_scenario = 0

                # Re-select clients
                if current_scenario == 'A':
                    selected_clients = select_clients_for_scenario(scenario_a_loaders, scenario_a_dataset)
                else:
                    selected_clients = select_clients_for_scenario(scenario_b_loaders, scenario_b_dataset)
                logging.info(f"[Concept Drift] Switched scenario to {current_scenario}, "
                             f"selected clients: {selected_clients}")

        # Get active loaders for the current scenario
        active_loaders = get_active_client_loaders()

        # Local training for selected clients
        client_updates = []
        round_losses, round_accs = [], []

        for client_id in selected_clients:
            local_model = CIFAR10_ResNet().to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate)

            # Train locally
            loss, acc = train_local_model(
                model=local_model,
                dataloader=active_loaders[client_id],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_local_epochs=args.num_local_epochs,
                frozen_layers=frozen_layers if args.enable_freezing else None
            )
            round_losses.append(loss)
            round_accs.append(acc)

            # Collect updates
            update_dict = get_client_update(local_model, frozen_layers if args.enable_freezing else None)
            client_updates.append(update_dict)

            # Log transmitted size
            tx_size_mb = get_transmitted_data_size(update_dict) / (1024 * 1024)
            logging.info(f"Client {client_id+1} => Update size: {tx_size_mb:.2f} MB")

        # Aggregate via FedAvg
        if client_updates:
            global_model = fed_avg(global_model, client_updates, 
                                   frozen_layers if args.enable_freezing else None)

        # Update stability & freeze/unfreeze if enabled
        if args.enable_freezing and client_updates:
            stability_dict = update_stability_smooth(global_model, client_updates, stability_dict, alpha=0.98)
            frozen_layers, cooldown_counters = apply_layer_freezing_with_hysteresis_and_cooldown(
                stability_dict,
                frozen_layers,
                cooldown_counters,
                freeze_threshold=args.freeze_threshold,
                unfreeze_threshold=args.unfreeze_threshold,
                cooldown=args.cooldown
            )
            log_freezing_status(frozen_layers, stability_dict)

        # Evaluate
        if args.enable_concept_drift:
            # Evaluate on scenario-specific test set to see localized performance
            active_test_loader = get_active_test_loader()
            test_acc = evaluate_global_model(global_model, active_test_loader, device)
            logging.info(f"Round {round_idx+1} => Test Accuracy (Scenario {current_scenario} classes only): {test_acc:.2f}%")
        else:
            # Evaluate on the full dataset
            test_acc = evaluate_global_model(global_model, test_loader_full, device)
            logging.info(f"Round {round_idx+1} => Test Accuracy (All classes): {test_acc:.2f}%")

        # Round-end info
        logging.info(f"Round {round_idx+1} => Avg Local Loss: {np.mean(round_losses):.4f}, "
                     f"Avg Local Acc: {np.mean(round_accs):.2f}%")
        elapsed = time.time() - start_time
        logging.info(f"Round {round_idx+1} completed in {elapsed:.2f}s\n")

        # Increment scenario round counter if concept drift is active
        if args.enable_concept_drift:
            rounds_in_current_scenario += 1

    logging.info("Federated learning process finished successfully.")

if __name__ == "__main__":
    main()
