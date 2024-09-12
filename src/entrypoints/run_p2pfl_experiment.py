import sys, time
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.communication.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from ..modelling.ernie import BERTLightningModel
import signal 

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
    comm = InMemoryCommunicationProtocol
    model = MLP
    data = MnistFederatedDM
    nr_nodes = 5
    nodes_refs: list[Node] = []
    for i in range(nr_nodes): 
        new_node = Node(model(),
                        data, 
                        f"mlp_test_{i}", 
                        protocol = comm)
        new_node.start(wait=False)
        nodes_refs.append(new_node)
    signal.signal(signal.SIGINT, lambda sig, frame: stop_nodes_handler(sig,frame,nodes_refs))
    print("test_nodes_started")
    ring(nodes=nodes_refs)