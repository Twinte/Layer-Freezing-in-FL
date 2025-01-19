# Federated Learning with Concept Drift Simulation

This repository implements a **Federated Learning (FL)** simulation using the CIFAR-10 dataset. It features:

- **Federated Averaging (FedAvg):** Standard aggregation of client models.
- **Layer Freezing:** Optional mechanism to freeze stable network layers to reduce communication overhead.
- **Concept Drift Simulation:** Optional simulation of changing data distributions between training rounds.
- **Non-IID Data Partitioning:** Uses a Dirichlet distribution to simulate non-independent and identically distributed (non-IID) data across clients.

The codebase is organized into separate modules for clarity and maintainability.

## Project Structure

```
federated_learning_project/
├── data_partition.py    # Data loading and non-IID partitioning functions
├── model.py             # CIFAR-10 ResNet model definition
├── client.py            # Client-side training and update functions
├── server.py            # Server-side aggregation, evaluation, stability, and freezing logic
└── main.py              # Main script orchestrating FL rounds and managing experiments
```

## Features

- **FedAvg:** Aggregates model updates from multiple clients.
- **Layer Freezing:** Dynamically freezes stable layers during training to reduce data transmission.
- **Concept Drift Simulation:** Switches between different class distributions (scenarios) over training rounds.
- **Toggleable Behaviors:** Easily enable/disable layer freezing and concept drift via command-line arguments.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy

Other standard libraries used include `argparse`, `logging`, etc.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Twinte/Layer-Freezing-in-FL.git
   cd Layer-Freezing-in-FL
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv env
   source env/bin/activate      # On Windows: env\Scripts\activate
   pip install torch torchvision numpy
   ```

## Usage

Run the main script with desired parameters. Below are some usage examples:

### Basic Federated Learning (no freezing, no concept drift)
```bash
python main.py --num_rounds 10 --alpha 0.1 --num_clients 30 --selection_rate 0.5
```

### Enable Layer Freezing
```bash
python main.py --enable_freezing --freeze_threshold 0.15 --unfreeze_threshold 0.25 --cooldown 5 \
               --num_rounds 10 --alpha 0.1 --num_clients 30 --selection_rate 0.5
```

### Enable Concept Drift
```bash
python main.py --enable_concept_drift \
               --scenario_a_classes 0,1,2,3,4 \
               --scenario_b_classes 5,6,7,8,9 \
               --concept_drift_switches 2 \
               --concept_drift_rounds_per_scenario 5 \
               --num_rounds 20 --alpha 0.1 --num_clients 30 --selection_rate 0.5
```

**Note:**
- When **concept drift** is enabled, the simulation alternates between two scenarios:
  - **Scenario A:** Training on a subset of classes (e.g., 0–4).
  - **Scenario B:** Training on a different subset (e.g., 5–9).
- The model evaluation during concept drift is **localized** to the currently active scenario's classes.
- When concept drift is disabled, evaluation is performed on the **entire** CIFAR-10 test set.

## Command-Line Arguments

- `--num_clients`: Number of clients (default: 30).
- `--selection_rate`: Fraction of clients selected per round (default: 0.5).
- `--alpha`: Dirichlet parameter for non-IID data partitioning (default: 0.1).
- `--learning_rate`: Learning rate for local training (default: 0.01).
- `--num_rounds`: Total number of global training rounds (default: 10).
- `--num_local_epochs`: Local epochs per client each round (default: 1).
- `--batch_size`: Batch size for training and testing (default: 64).
- `--enable_freezing`: Enable dynamic layer freezing (flag).
- `--freeze_threshold`: Stability threshold to freeze a layer (default: 0.15).
- `--unfreeze_threshold`: Stability threshold to unfreeze a layer (default: 0.25).
- `--cooldown`: Minimum rounds to wait before toggling freeze/unfreeze for a layer (default: 5).
- `--enable_concept_drift`: Enable concept drift simulation (flag).
- `--scenario_a_classes`: Comma-separated class indices for Scenario A (default: "0,1,2,3,4").
- `--scenario_b_classes`: Comma-separated class indices for Scenario B (default: "5,6,7,8,9").
- `--concept_drift_switches`: Number of times to switch between scenarios (default: 1).
- `--concept_drift_rounds_per_scenario`: Number of rounds to train before switching scenarios (default: 3).
- `--data_dir`: Directory for dataset download/storage (default: "./data").
- `--log_dir`: Directory to save log files (default: "./experiment_logs").
- `--seed`: Random seed for reproducibility (default: 42).

## Example Workflow

1. **Without Concept Drift:** 
   - The model is trained using standard FedAvg, optionally with layer freezing.
   - Evaluation covers the full CIFAR-10 test set.

2. **With Concept Drift:**
   - The system alternates between training on different subsets of classes.
   - Evaluation is localized to the classes in the active scenario, allowing analysis of scenario-specific performance and concept drift effects.

## License

This project is licensed under the MIT License.
