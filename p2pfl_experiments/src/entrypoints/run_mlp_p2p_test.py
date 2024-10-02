import matplotlib.pyplot as plt

from p2pfl.communication.grpc.grpc_communication_protocol import (
    GrpcCommunicationProtocol,
)
from p2pfl.communication.memory.memory_communication_protocol import (
    InMemoryCommunicationProtocol,
)
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
import time
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.utils import (
    wait_4_results,
    wait_convergence,
    set_test_settings
)
from p2pfl.settings import Settings
Settings.GRPC_TIMEOUT = 0.5
Settings.HEARTBEAT_PERIOD = 5
Settings.HEARTBEAT_TIMEOUT = 40
Settings.GOSSIP_PERIOD = 1
Settings.TTL = 40
Settings.GOSSIP_MESSAGES_PER_PERIOD = 9999999999
Settings.AMOUNT_LAST_MESSAGES_SAVED = 10000
Settings.GOSSIP_MODELS_PERIOD = 1
Settings.GOSSIP_MODELS_PER_ROUND = 4
Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 100
Settings.TRAIN_SET_SIZE = 4
Settings.VOTE_TIMEOUT = 60
Settings.AGGREGATION_TIMEOUT = 60
Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.2 * Settings.HEARTBEAT_TIMEOUT
Settings.LOG_LEVEL = "INFO"
Settings.EXCLUDE_BEAT_LOGS = True

def mnist(
    r: int,
    e: int,
    show_metrics: bool = True,
    measure_time: bool = False,
    use_local_protocol: bool = False,
) -> None:
    """
    P2PFL MNIST experiment.

    Args:
        n: The number of nodes.
        r: The number of rounds.
        e: The number of epochs.
        show_metrics: Show metrics.
        measure_time: Measure time.
        use_unix_socket: Use Unix socket.
        use_local_protocol: Use local protocol

    """
    if measure_time:
        start_time = time.time()

    # Node Creation
    nodes = []
    n = 50
    for i in range(n):
        address = f"node-{i}"
        node = Node(
            MLP(),
            MnistFederatedDM(sub_id=0, number_sub=20),  # sampling for increase speed
            protocol=InMemoryCommunicationProtocol,  # type: ignore
            address=address,
        )
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect(nodes[i].addr)
        time.sleep(0.1)
    wait_convergence(nodes, n - 1, only_direct=False)

    # Start Learning
    nodes[0].set_start_learning(rounds=r, epochs=e)

    # Wait and check
    wait_4_results(nodes)

    # Local Logs
    if show_metrics:
        local_logs = logger.get_local_logs()
        if local_logs != {}:
            logs_l = list(local_logs.items())[0][1]
            #  Plot experiment metrics
            for round_num, round_metrics in logs_l.items():
                for node_name, node_metrics in round_metrics.items():
                    for metric, values in node_metrics.items():
                        x, y = zip(*values)
                        plt.plot(x, y, label=metric)
                        # Add a red point to the last data point
                        plt.scatter(x[-1], y[-1], color="red")
                        plt.title(f"Round {round_num} - {node_name}")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.show()

        # Global Logs
        global_logs = logger.get_global_logs()
        if global_logs != {}:
            logs_g = list(global_logs.items())[0][1]  # Accessing the nested dictionary directly
            # Plot experiment metrics
            for node_name, node_metrics in logs_g.items():
                for metric, values in node_metrics.items():
                    x, y = zip(*values)
                    plt.plot(x, y, label=metric)
                    # Add a red point to the last data point
                    plt.scatter(x[-1], y[-1], color="red")
                    plt.title(f"{node_name} - {metric}")
                    plt.xlabel("Epoch")
                    plt.ylabel(metric)
                    plt.legend()
                    plt.show()

    # Stop Nodes
    for node in nodes:
        node.stop()

    if measure_time:
        print("--- %s seconds ---" % (time.time() - start_time))
