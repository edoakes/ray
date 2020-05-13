import argparse
import json
import os
import socket
import time
import datetime

import ray

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arg-size", type=str, required=True, help="'small' or 'large'")
parser.add_argument(
    "--num-nodes",
    type=int,
    required=True,
    help="Number of nodes in the cluster")
parser.add_argument(
    "--multi-node",
    action="store_true",
    help="Whether to put drivers on separate nodes")
parser.add_argument(
    "--timeline", action="store_true", help="Whether to dump a timeline")

NODES_PER_DRIVER = 5
CHAIN_LENGTH = 10
TASKS_PER_NODE_PER_BATCH = 2000


def get_node_ids():
    my_ip = ".".join(socket.gethostname().split("-")[1:])
    node_ids = set()
    for resource in ray.available_resources():
        if "node" in resource and my_ip not in resource:
            node_ids.add(resource)
    return node_ids


def get_local_node_resource():
    my_ip = ".".join(socket.gethostname().split("-")[1:])
    addr = "node:{}".format(my_ip)
    return addr


def do_ray_init(arg):
    internal_config = {"record_ref_creation_sites": 0}
    if os.environ.get("CENTRALIZED", False):
        internal_config["centralized_owner"] = 1
    elif os.environ.get("BY_VAL_ONLY", False):
        # Set threshold to 1 TiB to force everything to be inlined.
        internal_config["max_direct_call_object_size"] = 1024**4
        internal_config["max_grpc_message_size"] = -1
    else:
        # Base ownership case.
        internal_config.update({
            "initial_reconstruction_timeout_milliseconds": 100000,
            "num_heartbeats_timeout": 10,
            "lineage_pinning_enabled": 1,
            "free_objects_period_milliseconds": -1,
            "object_manager_repeated_push_delay_ms": 1000,
            "task_retry_delay_ms": 1000,
        })

    internal_config = json.dumps(internal_config)
    if os.environ.get("RAY_0_7", False):
        internal_config = None

    print("Starting ray with:", internal_config)
    ray.init(address="auto", _internal_config=internal_config)


def timeit(fn, trials=1, multiplier=1):
    start = time.time()
    for _ in range(1):
        start = time.time()
        fn()
        print("finished warmup iteration in", time.time() - start)

    stats = []
    for i in range(trials):
        start = time.time()
        fn()
        end = time.time()
        print("finished {}/{} in {}".format(i + 1, trials, end - start))
        stats.append(multiplier / (end - start))
        print("\tthroughput:", stats[-1])
    print("avg per second", round(np.mean(stats), 2), "+-",
          round(np.std(stats), 2))


@ray.remote
def f_small(*args):
    return b"hi"


@ray.remote
def f_large(*args):
    return np.zeros(1 * 1024 * 1024, dtype=np.uint8)


def do_batch(use_small, node_ids, args=None):
    if args is None:
        args = {}
        for node_id in node_ids:
            args[node_id] = None

    if use_small:
        f = f_small
    else:
        f = f_large

    results = dict()
    for node_id in node_ids:
        f_node = f.options(resources={node_id: 0.0001})

        batch = [
            f_node.remote(args[node_id])
            for _ in range(TASKS_PER_NODE_PER_BATCH)
        ]
        results[node_id] = f_node.remote(*batch)

    return results


def main(opts):
    do_ray_init(opts)

    node_ids = get_node_ids()
    assert opts.num_nodes % NODES_PER_DRIVER == 0
    num_drivers = int(opts.num_nodes / NODES_PER_DRIVER)
    while len(node_ids) < opts.num_nodes + num_drivers:
        print("{} / {} nodes have joined, sleeping for 1s...".format(
            len(node_ids), opts.num_nodes + num_drivers))
        time.sleep(1)
        node_ids = get_node_ids()

    print("All {} nodes joined: {}".format(len(node_ids), node_ids))
    worker_node_ids = list(node_ids)[:opts.num_nodes]
    driver_node_ids = list(node_ids)[opts.num_nodes:opts.num_nodes +
                                     num_drivers]

    use_small = args.arg_size == "small"

    @ray.remote(num_cpus=4 if opts.multi_node else 0)
    class Driver:
        def __init__(self, node_ids):
            # print("Driver starting with nodes:", node_ids)
            self.node_ids = node_ids

        def do_batch(self):
            prev = None
            for _ in range(CHAIN_LENGTH):
                prev = do_batch(use_small, self.node_ids, args=prev)

            ray.get(list(prev.values()))

        def ready(self):
            pass

    drivers = []
    for i in range(num_drivers):
        if opts.multi_node:
            node_id = driver_node_ids[i]
        else:
            node_id = get_local_node_resource()
        resources = {node_id: 0.001}
        worker_nodes = worker_node_ids[i * NODES_PER_DRIVER:(i + 1) *
                                       NODES_PER_DRIVER]
        drivers.append(
            Driver.options(resources=resources).remote(worker_nodes))

    ray.get([driver.ready.remote() for driver in drivers])

    def job():
        ray.get([driver.do_batch.remote() for driver in drivers])

    timeit(
        job,
        multiplier=len(worker_node_ids) * TASKS_PER_NODE_PER_BATCH *
        CHAIN_LENGTH)

    if opts.timeline:
        now = datetime.datetime.now()
        ray.timeline(filename="dump {}.json".format(now))


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.arg_size == "small" or args.arg_size == "large"
    main(args)
