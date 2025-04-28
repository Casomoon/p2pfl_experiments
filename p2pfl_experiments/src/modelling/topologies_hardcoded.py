import time
import random
from p2pfl.node import Node
from typing import Callable

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
        