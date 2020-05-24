import ray
import os
import numpy as np
import time


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self):
        pass

    def id(self):
        return os.uname()[1]

    def ping(self, data):
        return 0


this_ip = os.uname()[1]
remote_actor = None
ray.init(address="auto")

actors = [Actor.remote() for _ in range(20)]
for a in actors:
    other_ip = ray.get(a.id.remote())
    if other_ip != this_ip:
        remote_actor = a
        print("Found remote actor", other_ip, this_ip)
        break

for n in [
        1,
        1 * 1024,
        5 * 1024,
        10 * 1024,
        50 * 1024,
        100 * 1024,
        500 * 1024,
        1 * 1024 * 1024,
        5 * 1024 * 1024,
        10 * 1024 * 1024,
        50 * 1024 * 1024,
        100 * 1024 * 1024]:

    data = np.zeros(n, dtype=np.uint8)

    print("N =", n)
    trials = []
    for _ in range(4):
        start = time.time()
        ray.get([remote_actor.ping.remote(data) for _ in range(1000)])
        trial = 1000 / (time.time() - start)
        print("Mean throughput", trial)
        trials.append(trial)
    print("Average:", sum(trials[1:])/(len(trials)-1))
    print()
