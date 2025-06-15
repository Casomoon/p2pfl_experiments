# 🧪 Decentralized Federated Contradiction Detection

This repository contains a simulation setup for **fully decentralized federated learning** using the [`p2pfl`](https://github.com/danielp2pfl/p2pfl) framework. The experiment is designed to train a BERT model on the [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) dataset, with decentralized communication and aggregation using **gossip protocols**.

## ⚙️ Note on p2pfl

This project is based on `p2pfl` version 0.4, with **minor local modifications** (e.g., extended experiment, fixing of experiment start issues, changes to thread handling, additional logging etc.).  
Because of these changes, the `p2pfl` source code is included directly in this repository under the `p2pfl/` directory rather than installed via PyPI.

## ✅ Features

- Fully configurable experiment via command-line arguments
- Support for various network topologies: `ring`, `mesh`, `wheel`, `fully_connected`
- IID and non-IID (NIID) data distributions
- Gossip-based model dissemination (TTL, frequency, message limits)
- Deterministic training with reproducibility
- Automatically created result folders and logging system

---

## 📦 Setup

### Requirements

- Python == 3.9.20
- Poetry (https://python-poetry.org/)
- CUDA-capable GPU recommended for BERT training

# Install dependencies
poetry install

## 📁 Dataset Setup

Download and extract the [MultiNLI dataset](https://cims.nyu.edu/~sbowman/multinli/) into the following directory: <project-root>/data/multinli_1.0/
Make sure the folder contains files like `train.jsonl`, `dev_matched.jsonl`, etc.

## 🖥️ CLI Commands

This project defines convenient CLI entrypoints via Poetry:

| Command                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `poetry run run_bert_p2p`     | Run the decentralized P2P BERT experiment       |
| `poetry run run_bert_single`  | Run a single-node baseline BERT training         |
| `poetry run visualize_p2p`    | Visualize results from a P2P experiment          |
| `poetry run visualize_single` | Visualize results from the single-node baseline  |

## ⚙️ Configuration

All experiment settings are passed via command-line arguments:

| Argument                        | Description                                                  |
|--------------------------------|--------------------------------------------------------------|
| `--model_name`                 | e.g. `bert-base-uncased`                                     |
| `--structure`                  | Network topology: `ring`, `mesh`, `wheel`, `fully_connected`|
| `--nr_nodes`                   | Number of total nodes in the simulation                      |
| `--nr_learners`                | Number of active learners                                    |
| `--rounds`                     | Number of federated learning rounds                          |
| `--epochs_per_round`          | Local epochs per round                                       |
| `--batch_size`                | Batch size for training                                      |
| `--optimizer`                 | Optimizer used (e.g., `AdamW`, `SGD`)                        |
| `--learning_rate`             | Learning rate                                                |
| `--gossip_models_per_round`   | Number of models gossiped per round                          |
| `--gossip_models_period`      | Period (in seconds) between gossiping                        |
| `--gossip_messages_per_period`| Number of gossip messages per period                         |
| `--gossip_ttl`                | Time-To-Live for gossip messages                             |
| `--niid_data_amount`          | `True` for non-IID data distribution                         |

---

## ▶️ Running an Experiment

Example command to launch a run:

```bash
poetry run python scripts/experiments/run_gossip_bert.py \
    --model_name bert-base-uncased \
    --structure ring \
    --nr_nodes 17 \
    --nr_learners 17 \
    --rounds 15 \
    --epochs_per_round 1 \
    --batch_size 8 \
    --optimizer AdamW \
    --learning_rate 2e-5 \
    --gossip_models_per_round 4 \
    --gossip_models_period 5 \
    --gossip_messages_per_period 75 \
    --gossip_ttl 4 \
    --niid_data_amount True
```
## 🧪 Output Structure

Each run automatically creates a results directory at: <project-root>/run_results/<EXPERIMENT_NAME>/
The experiment name is dynamically generated from the key parameters. For example:
bert-ring-17-15-1-GOSS_75_4_5_4_NIID_DATA_DIST_True/
All logs and intermediate results are saved in this folder.
---
## 🔬 Reproducibility

This project ensures deterministic training by applying fixed seeds:
- Python `random.seed`
- NumPy seed
- PyTorch seed (CPU & CUDA)
- HuggingFace Transformers seed
- Deterministic CuDNN configuration (no benchmarking)
---
## 🧠 Notes
- For `ring` topologies, a low `gossip_ttl` is recommended.
- The `--niid_data_amount` flag enables realistic, uneven data splits across nodes.
- If a folder with the same experiment name already exists, the script will fail — either delete the old folder or modify parameters to generate a unique name.
---
## 📚 Related Work
This setup builds on research into contradiction detection using BERT in a decentralized federated learning setting with gossip-based communication. <Paper refernce HERE when published!!! > 
If you use this setup in your research, please cite appropriately.