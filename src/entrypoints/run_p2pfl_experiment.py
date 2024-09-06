import sys, time
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
from p2pfl.communication.memory.memory_communication_protocol import InMemoryCommunicationProtocol


def signal_handler(peers: list[Node]): 
    sim_log.warning(f"Received Interrupt ! Shutting down all nodes gracefully")
    for peer in peers : 
        peer.context.term()
    sys.exit()

def fully_connected(nodes : list[Node]):
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                nodes[i].connect(nodes[j].addr)
                time.sleep(0.1) 

def main(): 
    comm = InMemoryCommunicationProtocol
    model = MLP
    data = MnistFederatedDM
    nr_nodes = 5
    nodes_refs: list[Node] = []
    for i in range(nr_nodes): 
        new_node = Node(model, data, f"mlp_test_{i}", protocol = comm)
        new_node.start(wait=True)
        nodes_refs.append()
    print("test_nodes_started")
    