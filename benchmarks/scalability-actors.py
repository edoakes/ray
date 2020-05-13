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
    "--arg-size", type=str, default="small", help="'small' or 'large'")
parser.add_argument(
    "--num-nodes",
    type=int,
    required=True,
    help="Number of nodes in the cluster")
parser.add_argument(
    "--no-args", action="store_true", help="Submit tasks with no arguments")
parser.add_argument(
    "--sharded", action="store_true", help="Whether to shard the driver")
parser.add_argument(
    "--timeline", action="store_true", help="Whether to dump a timeline")

WORKER_CPUS = 4
NUM_DRIVERS = 10
TASKS_PER_NODE_PER_BATCH = 100000


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
            "initial_reconstruction_timeout_milliseconds": 100,
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


def timeit(fn, trials=5, multiplier=1):
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
class Actor:
    def f(self):
        return b"ok"


def main(opts):
    do_ray_init(opts)

    node_ids = get_node_ids()
    while len(node_ids) < opts.num_nodes:
        print("{} / {} have joined, sleeping for 1s...".format(
            len(node_ids), opts.num_nodes))
        time.sleep(1)
        node_ids = get_node_ids()
    node_ids = list(node_ids)[:opts.num_nodes]
    print("All {} nodes joined: {}".format(len(node_ids), node_ids))

    @ray.remote(num_cpus=0, resources={get_local_node_resource(): 0.0001})
    class Driver:
        def __init__(self, node_ids):
            self.my_actors = []
            for node_id in node_ids:
                for _ in range(WORKER_CPUS):
                    self.my_actors.append(
                        Actor.options(resources={
                            node_id: 0.0001
                        }).remote())

        def do_batch(self, tasks_per_node):
            results = []
            tasks_per_core = int(tasks_per_node / WORKER_CPUS)
            for idx in range(0, tasks_per_node, tasks_per_core):
                actor = self.my_actors[int(idx / tasks_per_core)]
                results.extend(
                    [actor.f.remote() for _ in range(tasks_per_core)])
            ray.get(results)

    assert len(node_ids) % NUM_DRIVERS == 0
    assert TASKS_PER_NODE_PER_BATCH % WORKER_CPUS == 0
    nodes_per_driver = int(len(node_ids) / NUM_DRIVERS)

    drivers = []
    for i in range(NUM_DRIVERS):
        nodes = node_ids[i * nodes_per_driver:(i + 1) * nodes_per_driver]
        drivers.append(Driver.remote(nodes))

    def job():
        ray.get([
            driver.do_batch.remote(TASKS_PER_NODE_PER_BATCH)
            for driver in drivers
        ])

    timeit(job, multiplier=len(node_ids) * TASKS_PER_NODE_PER_BATCH)

    del drivers
    time.sleep(1)

    if opts.timeline:
        now = datetime.datetime.now()
        ray.timeline(filename="dump {}.json".format(now))


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.arg_size == "small" or args.arg_size == "large"
    main(args)
