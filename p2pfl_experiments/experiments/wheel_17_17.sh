#!/bin/bash
cd ~/p2pfl_experiments/p2pfl_experiments
pwd
model_name="bert"
structure="wheel"
nr_nodes=17
nr_learners=17
epochs_per_round=1
rounds=4
batch_size=1
#data_dist_weights=(0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555 0.05555555)
# gossip settings
gossip_models_per_round=5
gossip_models_period=2
gossip_messages_per_period=5
gossip_ttl=15
# niid
niid_data_amount=True

srun poetry run run_bert_p2p \
        --model_name "$model_name"\
        --structure "$structure"\
        --nr_nodes "$nr_nodes"\
        --nr_learners "$nr_learners"\
        --epochs_per_round "$epochs_per_round"\
        --rounds "$rounds"\
        --batch_size "$batch_size"\
        --gossip_models_per_round "$gossip_models_per_round" \
        --gossip_models_period "$gossip_models_period"\
        --gossip_messages_per_period "$gossip_messages_per_period"\
        --gossip_ttl "$gossip_ttl"\
        --niid_data_amount "$niid_data_amount"