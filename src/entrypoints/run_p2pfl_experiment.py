import sys, time
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.communication.memory.memory_communication_protocol import InMemoryCommunicationProtocol
# singleton logger
from p2pfl.management.logger import logger
from ..modelling.ernie import BERTLightningModel
from ..modelling.nli_data_load import NLIDataModule, NLIParser
from pathlib import Path
import signal 

root = Path(__file__).resolve().parents[2]
mnli_data_path = root/"data"/"multinli_1.0"

def stop_nodes_handler(sig, frame, nodes: list[Node]):
    print(f"Received signal {sig}. Stopping all nodes...")
    for node in nodes:
        node.stop()  
    sys.exit(0)

def fully_connected(nodes: list[Node]):
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            nodes[i].connect(nodes[j].addr)
            time.sleep(0.1) 
    print("fully_connected")

def ring(nodes: list[Node]): 
    num_nodes = len(nodes)
    for i in range(num_nodes): 
        nodes[i].connect(nodes[(i+1)%num_nodes].addr)
        time.sleep(0.1)
    print("circle_merkel")

def main(): 
    logger.info("main", f"Extracting mnli data from {mnli_data_path}.")
    comm = InMemoryCommunicationProtocol
    model = MLP
    nr_nodes = 5
    data_dist_weights = [0.1, 0.15, 0.15, 0.3, 0.3]
    batch_size = 16
    assert len(data_dist_weights) == nr_nodes
    assert sum(data_dist_weights) == 1.0
    nodes_refs: list[Node] = []
    nli_data = NLIParser(mnli_data_path, nr_nodes, data_dist_weights, batch_size).get_non_iid_split()
    exit(1)
    
    
    
    for i in range(nr_nodes): 
        new_node = Node(model(),
                        data, 
                        f"mlp_test_{i}", 
                        protocol = comm)
        new_node.start()
        nodes_refs.append(new_node)
    signal.signal(signal.SIGINT, lambda sig, frame: stop_nodes_handler(sig,frame,nodes_refs))
    print("test_nodes_started")
    ring(nodes=nodes_refs)