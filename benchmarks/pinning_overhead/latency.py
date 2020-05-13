import json
import time
import socket
import os

import ray

import numpy as np

NUM_TRIALS = 100
CONDITIONS = [0, 1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024]

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

def sizeof_fmt(num):
    for unit in ["", "Ki", "Mi", "Gi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, "B")
        num /= 1024.0

    return "%.1f%s%s" % (num, "Yi", "B")


@ray.remote(num_cpus=4)
class Actor2:
    def __init__(self, obj_size):
        self.obj_size = obj_size

    @ray.method(num_return_vals=2)
    def get(self):
        return np.zeros(self.obj_size, dtype=np.uint8), b"ok"


@ray.remote(num_cpus=0, resources={get_local_node_resource(): 0.001})
class Actor1:
    def __init__(self, other):
        self.other = other

    def run(self):
        times = []
        for trial in range(NUM_TRIALS + 10):
            start = time.time()
            _, ok = self.other.get.remote()
            ray.get(ok)
            trial_time = time.time() - start
            if trial >= 10:
                times.append(trial_time)
            del _, ok
            time.sleep(0.1)

        return times


def run_trial(obj_size):
    actor2 = Actor2.remote(obj_size)
    actor1 = Actor1.remote(actor2)

    times = ray.get(actor1.run.remote())
    assert len(times) == NUM_TRIALS
    print("\t", sizeof_fmt(obj_size), "avg (ms):", 1000*(sum(times) / len(times)))


def main():
    config = {"record_ref_creation_sites": 0}
    if "RAY_PINNING_DISABLED" in os.environ:
        config["object_pinning_enabled"] = 0
    print("Starting ray with:", config)
    ray.init(address="auto", _internal_config=json.dumps(config))
    for obj_size in CONDITIONS:
        run_trial(obj_size)

if __name__ == "__main__":
    main()
