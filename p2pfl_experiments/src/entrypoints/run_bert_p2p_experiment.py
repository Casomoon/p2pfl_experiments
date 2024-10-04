
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import sys, time
from p2pfl.node import Node
from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.settings import Settings
from p2pfl.utils import wait_to_finish
# singleton logger
from p2pfl.management.logger import logger
from ..modelling.bert_lightning import BERTLightningModel
from ..modelling.nli_data_load import NLIParser
from pathlib import Path
import signal 
import sys
import random

root = Path(__file__).resolve().parents[2]
mnli_data_path = root/"data"/"multinli_1.0"

STRUCTURE = "multi_star"
NR_NODES = 20
DATA_DIST_WEIGHTS = [0.04842105, 0.04842105, 0.03842105, 0.03842105, 0.04842105,
    0.04842105, 0.03842105, 0.03842105, 0.03842105, 0.04842105,
    0.04842105, 0.03842105, 0.04842105, 0.04842105, 0.03842105,
    0.04842105, 0.03842105, 0.06842105, 0.03842105, 0.15000005]
ROUNDS = 5
EPOCHS_PER_ROUND = 2
BATCH_SIZE = 1
# graceful stopping of the training nodes 
def stop_nodes_handler(sig, frame, nodes: list[Node]):
    print(f"Received signal {sig}. Stopping all nodes...")
    for node in nodes:
        node.stop()  
    sys.exit(0)

# fully connect all nodes in the network 
def fully_connected(nodes: list[Node]):
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            nodes[i].connect(nodes[j].addr)
            time.sleep(0.1) 
    print("fully_connected")
# create a ring structure
def ring(nodes: list[Node]): 
    num_nodes = len(nodes)
    for i in range(num_nodes): 
        nodes[i].connect(nodes[(i+1)%num_nodes].addr)
        time.sleep(0.1)

def star(nodes: list[Node]): 
    num_nodes = len(nodes)
    central_node = nodes[0]  # Select the first node as the central node
    for i in range(1, num_nodes): 
        central_node.connect(nodes[i].addr)
        time.sleep(0.1)
    print("star")

def mesh(nodes: list[Node], connection_prob=0.5):  # 50% chance to connect any two nodes
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < connection_prob:  # connect with a probability
                nodes[i].connect(nodes[j].addr)
                time.sleep(0.1)
    print("mesh")   

def multi_star(nodes: list[Node], num_connectors: int): 
    """
    Creates a multi-level star topology where 'num_connectors' nodes act as the hubs.
    
    Args:
    nodes (list[Node]): The list of all nodes in the network.
    num_connectors (int): The number of connector nodes.
    """
    num_nodes = len(nodes)
    
    # The first 'num_connectors' nodes are considered as connectors
    connectors = nodes[:num_connectors]
    
    # Connect all non-connector nodes to one of the connector nodes
    for i in range(num_connectors, num_nodes): 
        connector = connectors[i % num_connectors]  # Distribute connections among connectors
        connector.connect(nodes[i].addr)
        time.sleep(0.1)
    
    # interconnect the connector nodes to each other
    for i in range(num_connectors):
        for j in range(i + 1, num_connectors):
            connectors[i].connect(connectors[j].addr)
            time.sleep(0.1)
    
    print("multi_star")

def wait_n_neigh(nodes: list[Node], n_neis: int, wait: int = 60, only_direct: bool = False): 
    acum = 0.0
    while True:
        begin = time.time()
        if all(len(n.get_neighbors(only_direct=only_direct)) == n_neis for n in nodes):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        logger.info("main",f"Time waited:{acum}")
        if acum > wait:
            raise AssertionError()
        
def set_struct_dict(): pass
def set_test_settings() -> None:
    """Set settings for testing."""
    Settings.GRPC_TIMEOUT = 0.5
    Settings.HEARTBEAT_PERIOD = 30
    Settings.HEARTBEAT_TIMEOUT = 4500
    Settings.GOSSIP_PERIOD = 5
    Settings.TTL = 40
    Settings.GOSSIP_MESSAGES_PER_PERIOD = 9999999999
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 10000
    Settings.GOSSIP_MODELS_PERIOD = 5
    Settings.GOSSIP_MODELS_PER_ROUND = 50
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 50
    Settings.TRAIN_SET_SIZE = 4
    Settings.VOTE_TIMEOUT = 4000
    Settings.AGGREGATION_TIMEOUT = 10000
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.05 * Settings.HEARTBEAT_TIMEOUT
    Settings.LOG_LEVEL = "DEBUG"


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
    logger.info("main", f"ROUNDS : {ROUNDS}")
    logger.info("main", f"EPOCHS_PER_ROUND : {EPOCHS_PER_ROUND}")
def main(): 
    logger.__name__
    torch.set_float32_matmul_precision("medium")
    set_test_settings()
    log_run_settings()
    logger.info("main", f"Extracting mnli data from {mnli_data_path}.")
    comm = InMemoryCommunicationProtocol
    bert_model_init = BERTLightningModel
    #nr_nodes = 2
   
    nodes_refs: list[Node] = []
    # create the data distribution
    nli_data_parser = NLIParser(mnli_data_path, NR_NODES, DATA_DIST_WEIGHTS, BATCH_SIZE)
    # prepare the data split initially 
    data_modules = nli_data_parser.get_non_iid_split()
     
    for i in range(NR_NODES): 
        # wrap it into Lightning data modules
        # create the nodes
        new_node = Node(bert_model_init(cid=0, model_name='bert-base-uncased', num_labels=2, lr=2e-5),
                        data_modules[i], 
                        f"BERT_{i}", 
                        protocol = comm)
        new_node.start()
        nodes_refs.append(new_node)
    # graceful stopping
    signal.signal(signal.SIGINT, lambda sig, frame: stop_nodes_handler(sig,frame,nodes_refs))
    print("test_nodes_started")
    # fully connected network    
    multi_star(nodes_refs, 3)
    wait_n_neigh(nodes_refs,NR_NODES - 1, only_direct=False)
    nodes_refs[0].set_start_learning(rounds = ROUNDS, epochs = EPOCHS_PER_ROUND)
    one_day_in_sec = 86400 
    wait_to_finish(nodes_refs, one_day_in_sec)