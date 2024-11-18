#base
MODEL_NAME: str 
STRUCTURE: str
NR_NODES: int
NR_LEARNERS: int
DATA_DIST_WEIGHTS: list[int]
ROUNDS : int 
EPOCHS_PER_ROUND: int 
BATCH_SIZE: int 
EXPERIMENT_NAME: str
# gossip 
GOSSIP_MODELS_PER_ROUND: int
GOSSIP_MODELS_PERIOD: int
GOSSIP_MESSAGES_PER_PERIOD: int
from p2pfl.settings import Settings
import argparse
import math
# call the overwrite of Settings before anything else
def parse_args(): 
    # base params
    global MODEL_NAME, STRUCTURE, NR_NODES, NR_LEARNERS, ROUNDS, EPOCHS_PER_ROUND, BATCH_SIZE, DATA_DIST_WEIGHTS, EXPERIMENT_NAME
    # gossip params
    global GOSSIP_MODELS_PER_ROUND, GOSSIP_MODELS_PERIOD, GOSSIP_MESSAGES_PER_PERIOD
    parser = argparse.ArgumentParser(description="Run P2P FL with customizable parameters")
    # base args for the federation
    parser.add_argument("--model_name", type=str, default="bert", help="Model name")
    parser.add_argument("--structure", type=str, choices=["fully_connected", "ring", "star", "mesh", "multi_star"], default="multi_star", help="Network structure")
    parser.add_argument("--nr_nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--nr_learners", type=int, default=20, help="Number of learners")
    parser.add_argument("--rounds", type=int, default=10, help="Number of training rounds")
    parser.add_argument("--epochs_per_round", type=int, default=1, help="Epochs per round")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--connection_prob", type=float, default=0.5, help="Connection probability for mesh structure")
    parser.add_argument("--data_dist_weights", type=float, nargs='+', default=[0.12, 0.08, 0.15, 0.10, 0.07, 0.05, 0.09, 0.11, 0.06, 0.10, 0.07], help="Data distribution weights")
    # gossip parameters  
    parser.add_argument("--gossip_models_per_round", type=int, default = 4)
    parser.add_argument("--gossip_models_period", type=int, default = 5)
    parser.add_argument("--gossip_messages_per_period", type=int, default=75)
    # get it 
    args = parser.parse_args()
    assert args.nr_learners<=args.nr_nodes
    # set it 
    MODEL_NAME = args.model_name
    STRUCTURE = args.structure
    NR_NODES = args.nr_nodes
    NR_LEARNERS = args.nr_learners
    ROUNDS = args.rounds
    EPOCHS_PER_ROUND = args.epochs_per_round
    BATCH_SIZE = args.batch_size
    DATA_DIST_WEIGHTS = args.data_dist_weights
    print(sum(DATA_DIST_WEIGHTS))
    assert math.isclose(sum(DATA_DIST_WEIGHTS),1.0,rel_tol=1e-5)
    
    GOSSIP_MESSAGES_PER_PERIOD = args.gossip_messages_per_period
    GOSSIP_MODELS_PER_ROUND = args.gossip_models_per_round
    GOSSIP_MODELS_PERIOD = args.gossip_models_period

    # Set EXPERIMENT_NAME based on the values provided
    EXPERIMENT_NAME = f"{MODEL_NAME}_{STRUCTURE}_{NR_NODES}_{ROUNDS}_{EPOCHS_PER_ROUND}_GOSS_{GOSSIP_MESSAGES_PER_PERIOD}_{GOSSIP_MODELS_PER_ROUND}_{GOSSIP_MODELS_PERIOD}"
def set_test_settings() -> None:
    """Set settings for testing."""
    Settings.GRPC_TIMEOUT = 0.5
    Settings.HEARTBEAT_PERIOD = 30
    Settings.HEARTBEAT_TIMEOUT = 4500
    Settings.GOSSIP_PERIOD = 5
    # FOR RING LOWEST TTL POSSIBLE
    Settings.TTL = 15
    Settings.GOSSIP_MESSAGES_PER_PERIOD = GOSSIP_MESSAGES_PER_PERIOD
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 100
    Settings.GOSSIP_MODELS_PERIOD = GOSSIP_MODELS_PERIOD
    Settings.GOSSIP_MODELS_PER_ROUND = GOSSIP_MODELS_PER_ROUND
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 10
    Settings.TRAIN_SET_SIZE = NR_LEARNERS
    Settings.VOTE_TIMEOUT = 4000
    Settings.AGGREGATION_TIMEOUT = 30000
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.1 * Settings.HEARTBEAT_TIMEOUT
    Settings.LOG_LEVEL = "DEBUG"
parse_args()
set_test_settings()

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import sys, time
import numpy as np
from p2pfl.node import Node
from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.learning.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.pytorch.lightning_model import LightningModel
from p2pfl.learning.aggregators.fedavg import FedAvg


from p2pfl.utils import wait_to_finish
# singleton logger
from p2pfl.management.logger import logger
from ..modelling.bert_lightning import BERTLightningModel
from ..modelling.nli_data_load import NLIParser
from pathlib import Path
import signal 
import sys

root = Path(__file__).resolve().parents[2]
mnli_data_path = root/"data"/"multinli_1.0"


# graceful stopping of the training nodes 
def stop_nodes_handler(sig, frame, nodes: list[Node]):
    print(f"Received signal {sig}. Stopping all nodes...")
    for node in nodes:
        node.stop()  
    sys.exit(0)


def wait_n_neigh(nodes: list[Node], n_neis: int, wait: int = 150, only_direct: bool = False): 
    acum = 0.0
    while True:
        begin = time.time()
        if all(len(n.get_neighbors(only_direct=only_direct)) >= n_neis for n in nodes):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        logger.info("main",f"Time waited:{acum}")
        if acum > wait:
            break
        
def log_run_settings()-> None: 
    logger.info("main", f"Settings.HEARTBEAT_PERIOD : {Settings.HEARTBEAT_PERIOD}")
    logger.info("main", f"Settings.HEARTBEAT_TIMEOUT : {Settings.HEARTBEAT_TIMEOUT}")
    logger.info("main", f"Settings.GOSSIP_PERIOD : {Settings.GOSSIP_PERIOD}")
    logger.info("main", f"Settings.TTL : {Settings.TTL}")
    logger.info("main", f"Settings.GOSSIP_MESSAGES_PER_PERIOD : {Settings.GOSSIP_MESSAGES_PER_PERIOD}")
    logger.info("main", f"Settings.AMOUNT_LAST_MESSAGES_SAVED : {Settings.AMOUNT_LAST_MESSAGES_SAVED}")
    logger.info("main", f"Settings.GOSSIP_MODELS_PERIOD : {Settings.GOSSIP_MODELS_PERIOD}")
    logger.info("main", f"Settings.GOSSIP_MODELS_PER_ROUND : {Settings.GOSSIP_MODELS_PER_ROUND}")
    logger.info("main", f"Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS : {Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS}")
    logger.info("main", f"Settings.AGGREGATION_TIMEOUT : {Settings.AGGREGATION_TIMEOUT}")
    logger.info("main", f"Settings.TRAIN_SET_SIZE : {Settings.TRAIN_SET_SIZE}")
    logger.info("main", f"Settings.VOTE_TIMEOUT : {Settings.VOTE_TIMEOUT}")
    logger.info("main", f"Settings.WAIT_HEARTBEATS_CONVERGENCE : {Settings.WAIT_HEARTBEATS_CONVERGENCE}")
    # Log the parsed arguments
    logger.info("main", f"MODEL_NAME : {MODEL_NAME}")
    logger.info("main", f"STRUCTURE : {STRUCTURE}")
    logger.info("main", f"NR_NODES : {NR_NODES}")
    logger.info("main", f"NR_LEARNERS : {NR_LEARNERS}")
    logger.info("main", f"DATA_DIST_WEIGHTS : {DATA_DIST_WEIGHTS}")
    logger.info("main", f"ROUNDS : {ROUNDS}")
    logger.info("main", f"EPOCHS_PER_ROUND : {EPOCHS_PER_ROUND}")
    logger.info("main", f"BATCH_SIZE : {BATCH_SIZE}")
    logger.info("main", f"EXPERIMENT_NAME : {EXPERIMENT_NAME}")

def setup_results_dir(): 
    base_results_dir: Path = Path(__file__).resolve().parents[2]/"run_results"
    if not base_results_dir.exists():
        base_results_dir.mkdir()
    run_results_dir = base_results_dir/f"{EXPERIMENT_NAME}"
    assert not run_results_dir.exists()
    run_results_dir.mkdir()
    return run_results_dir

def set_deterministic_training(seed: int ): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def log_gpu_settings():
    logger.info("main", f"CUDA availabru {torch.cuda.is_available()}")
    logger.info("main", f"CUDA GPU {torch.cuda.get_device_name(0)}")
def main(): 
    torch.set_float32_matmul_precision("medium")
    log_run_settings()
    set_deterministic_training(420)
    log_gpu_settings()
    model_init_blm = BERTLightningModel
    module_adapter : P2PFLModel = LightningModel
    nodes_refs: list[Node] = []
    # create the data distribution
    logger.info("main", f"Extracting mnli data from {mnli_data_path}.")
    nli_data_parser = NLIParser(mnli_data_path, NR_NODES, DATA_DIST_WEIGHTS, MODEL_NAME, BATCH_SIZE, overall_cut=0.00)
    # prepare the data split initially 
    data_modules = nli_data_parser.get_non_iid_split()
    # create the directory to drop off the results of the run during the run.
    run_dir = setup_results_dir()
    for i in range(NR_NODES): 
        # wrap it into Lightning data modules
        # create the nodes
        new_node = Node(model = module_adapter(model_init_blm(
                                                cid=i,
                                                model_name= MODEL_NAME,
                                                num_labels=2,
                                                lr=2e-5,
                                                base_dir=run_dir)),
                        data = data_modules[i], 
                        address = f"BERT_{i}", 
                        protocol = InMemoryCommunicationProtocol,
                        learner = LightningLearner,
                        aggregator = FedAvg)
        new_node.start()
        nodes_refs.append(new_node)
    # graceful stopping
    signal.signal(signal.SIGINT, lambda sig, frame: stop_nodes_handler(sig,frame,nodes_refs))
    from ..modelling.topologies_hardcoded import get_topology
    topology_function = get_topology(STRUCTURE)
    topology_function(nodes_refs)
    # only direct !, refer
    wait_n_neigh(nodes_refs, NR_NODES -1 , only_direct=True)
    nodes_refs[15].set_start_learning(rounds=ROUNDS, epochs = EPOCHS_PER_ROUND)
    logger.info("main", "15 started")
    #nodes_refs[0].set_start_learning(rounds = ROUNDS, epochs = EPOCHS_PER_ROUND)
    #logger.info("main", "0 started")
    #nodes_refs[6].set_start_learning(rounds = ROUNDS, epochs = EPOCHS_PER_ROUND)
    #logger.info("main", "6 started")
    one_day_in_sec = 86400 
    wait_to_finish(nodes_refs, one_day_in_sec)