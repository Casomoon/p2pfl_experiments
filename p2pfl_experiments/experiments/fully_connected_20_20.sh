#!/bin/bash
cd ~/p2pfl_experiments/p2pfl_experiments
pwd
model_name="bert"
structure="fully_connected"
nr_nodes=20
nr_learners=20
epochs_per_round=1
rounds=10
batch_size=8
data_dist_weights=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)

poetry run run_bert_p2p \
        --model_name "$model_name"\
        --structure "$structure"\
        --nr_nodes "$nr_nodes"\
        --nr_learners "$nr_learners"\
        --epochs_per_round "$epochs_per_round"\
        --rounds "$rounds" \
        --batch_size "$batch_size"\
        --data_dist_weights "${data_dist_weights[@]}"

