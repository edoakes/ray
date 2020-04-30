import json
import time

import ray

import numpy as np

CONDITIONS = [
    (0, 100),
    (1024, 10000),
    (100 * 1024, 10000),
    (1024 * 1024, 1000),
    (10 * 1024 * 1024, 1000),
    (100 * 1024 * 1024, 1000),
]


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


@ray.remote(num_cpus=4)
class Actor1:
    def __init__(self, other):
        self.other = other

    def run(self, num_trials):
        times = []
        for trial in range(num_trials + 100):
            start = time.time()
            obj = ray.get(self.other.get.remote())
            if trial >= 100:
                times.append(time.time() - start)
            del obj
            time.sleep(0.01)

        return times


def run_trial(obj_size, num_trials):
    actor2 = Actor2.remote(obj_size)
    actor1 = Actor1.remote(actor2)

    times = ray.get(actor1.run.remote(num_trials))
    assert len(times) == num_trials
    print("\t", sizeof_fmt(obj_size), "avg:", sum(times) / len(times))


def main():
    config = {"record_ref_creation_sites": 0}
    ray.init(_internal_config=json.dumps(config))
    print("Pinning:")
    for obj_size, num_trials in CONDITIONS:
        run_trial(obj_size, num_trials)

    print("")
    ray.shutdown()

    config["object_pinning_enabled"] = 0
    ray.init(_internal_config=json.dumps(config))
    print("No pinning:")
    for obj_size, num_trials in CONDITIONS:
        run_trial(obj_size, num_trials)


if __name__ == "__main__":
    main()
