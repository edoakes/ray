import time
import random
import json

import ray
conf = {"initial_reconstruction_timeout_milliseconds": 100}
if ray.__version__ != "0.7.7":
    conf["object_store_full_initial_delay_ms"] = "50"
ray.init(
    object_store_memory=100 * 1024 * 1024, _internal_config=json.dumps(conf))

import numpy as np

NUM_ITERATIONS = 50
UPDATE_SIZE = int(1 * 1024 * 1024 / 8)


def random_array():
    return np.random.rand(1, UPDATE_SIZE)


@ray.remote
def compute():
    return random_array()


@ray.remote
def update_batch(num_updates):
    # "read" some records, random size that sometimes fails.
    array_sum = random_array()
    updates = [compute.remote() for _ in range(num_updates)]
    while len(updates) > 0:
        [ready], updates = ray.wait(updates)
        array_sum += ray.get(ready)
    return array_sum / num_updates


@ray.remote
def do_update(current, update, i, iter_start):
    #print("Finished iter", i, "in ", time.time() - iter_start)
    return current + update


def main():
    # Keep updating one large working object.
    # Each iteration, pass it out to some other tasks
    # that create subtasks that generate tmp objects.
    results = []
    for num_updates in [20] + list(range(0, 220, 20)):
        start = time.time()
        current = compute.remote()
        for i in range(NUM_ITERATIONS):
            iter_start = time.time()
            # Combine updates.
            current = do_update.remote(current, update_batch.remote(num_updates), i,
                                       iter_start)
            ray.wait([current])

        results.append(time.time() - start)
        print(results[-1])
    print("results:")
    print(",".join([str(result) for result in results[1:]]))


if __name__ == "__main__":
    main()
