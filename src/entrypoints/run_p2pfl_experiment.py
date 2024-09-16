import sys, time
from p2pfl.node import Node
from p2pfl.communication.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.settings import Settings
from p2pfl.utils import (
    wait_4_results,
    wait_convergence,
)
# singleton logger
from p2pfl.management.logger import logger
from ..modelling.bert_lightning import BERTLightningModel
from ..modelling.nli_data_load import NLIDataModule, NLIParser
from pathlib import Path
import signal 

root = Path(__file__).resolve().parents[2]
mnli_data_path = root/"data"/"multinli_1.0"

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
    print("circle_merkel")

def wait_n_neigh(nodes: list[Node], n_neis: int, wait: int = 5, only_direct: bool = False): 
    acum = 0.0
    while True:
        begin = time.time()
        if all(len(n.get_neighbors(only_direct=only_direct)) == n_neis for n in nodes):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > wait:
            raise AssertionError()
def set_test_settings() -> None:
    """Set settings for testing."""
    Settings.GRPC_TIMEOUT = 0.5
    Settings.HEARTBEAT_PERIOD = 0.5
    Settings.HEARTBEAT_TIMEOUT = 2
    Settings.GOSSIP_PERIOD = 0
    Settings.TTL = 10
    Settings.GOSSIP_MESSAGES_PER_PERIOD = 100
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 100
    Settings.GOSSIP_MODELS_PERIOD = 1
    Settings.GOSSIP_MODELS_PER_ROUND = 4
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 4
    Settings.TRAIN_SET_SIZE = 4
    Settings.VOTE_TIMEOUT = 60
    Settings.AGGREGATION_TIMEOUT = 60
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.2 * Settings.HEARTBEAT_TIMEOUT
    Settings.LOG_LEVEL = "DEBUG"

def main(): 
    set_test_settings()
    logger.info("main", f"Extracting mnli data from {mnli_data_path}.")
    comm = InMemoryCommunicationProtocol
    model = BERTLightningModel
    #nr_nodes = 2
    #data_dist_weights = [0.5, 0.5]
    nr_nodes = 3
    data_dist_weights = [0.33, 0.33, 0.34]
    batch_size = 16
    assert len(data_dist_weights) == nr_nodes
    assert sum(data_dist_weights) == 1.0
    nodes_refs: list[Node] = []
    # create the data distribution
    nli_data_parser = NLIParser(mnli_data_path, nr_nodes, data_dist_weights, batch_size)
    # prepare the data split initially 
    nli_data_parser.get_non_iid_split()
     
    for i in range(nr_nodes): 
        # wrap it into Lightning data modules
        data_module = NLIDataModule(
            parser = nli_data_parser,
            cid = i, 
            niid = True
        )
        # create the nodes
        new_node = Node(model(),
                        data_module, 
                        f"BERT_{i}", 
                        protocol = comm)
        new_node.start()
        nodes_refs.append(new_node)
    # graceful stopping
    signal.signal(signal.SIGINT, lambda sig, frame: stop_nodes_handler(sig,frame,nodes_refs))
    print("test_nodes_started")
    # fully connected network    
    fully_connected(nodes_refs)
    wait_convergence(nodes_refs, nr_nodes - 1, only_direct=False)
    nodes_refs[0].set_start_learning(rounds = 5, epochs = 5)
    wait_4_results(nodes_refs)