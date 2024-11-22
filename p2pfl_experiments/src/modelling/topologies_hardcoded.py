import time
import random
from p2pfl.node import Node
from typing import Callable

#def mesh_full
#def mesh_90
#def mesh_60
#def mesh_30 
#def ring
#def wheel

def get_topology(top_str: str)->Callable: 
    topologies = {
        "fully_connected" : fully_connected, 
        "ring": ring, 
        "wheel": wheel

    }
    assert top_str in topologies, f"Unknown topology: '{top_str}'"
    top_func: Callable = topologies.get(top_str)
    return top_func

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

def wheel(nodes: list[Node]): 
    ring(nodes)
    num_nodes = len(nodes)
    even = (num_nodes%2==0) 

    for i in range(num_nodes//2):
        opposite_index = i+(num_nodes//2)%num_nodes
        nodes[i].connect(nodes[opposite_index].addr)
        time.sleep(0.1)
    if not even:
        # If the number of nodes is odd, connect the middle node to its nearest opposites
        middle_index = num_nodes // 2
        opposite_left = (middle_index - (num_nodes // 2)) % num_nodes
        opposite_right = (middle_index + (num_nodes // 2)) % num_nodes
        
        nodes[middle_index].connect(nodes[opposite_left].addr)
        nodes[middle_index].connect(nodes[opposite_right].addr)
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