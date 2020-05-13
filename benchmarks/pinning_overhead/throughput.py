import json
import time
import socket
import os

import ray

import numpy as np

NUM_TASKS = 5000
CONDITIONS = [0, 1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024]


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

    def get(self):
        return np.zeros(self.obj_size, dtype=np.uint8)


@ray.remote(num_cpus=0, resources={get_local_node_resource(): 0.001})
class Actor1:
    def __init__(self, other):
        self.other = other

    def run(self):
        for _ in range(NUM_TASKS):
            prev = self.other.get.remote()

        ray.get(prev)


def timeit(fn, trials=3, multiplier=1, name=""):
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
    print(name, "avg per second", round(np.mean(stats), 2), "+-",
          round(np.std(stats), 2))


def main():
    config = {"record_ref_creation_sites": 0}
    if "RAY_PINNING_DISABLED" in os.environ:
        config["object_pinning_enabled"] = 0
    print("Starting ray with:", config)
    ray.init(address="auto", _internal_config=json.dumps(config))
    for obj_size in CONDITIONS:
        actor2 = Actor2.remote(obj_size)
        actor1 = Actor1.remote(actor2)
        def trial():
            ray.get(actor1.run.remote())

        timeit(trial, multiplier=NUM_TASKS, name=sizeof_fmt(obj_size))

if __name__ == "__main__":
    main()
