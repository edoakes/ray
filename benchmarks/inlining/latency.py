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
        return data


this_ip = os.uname()[1]
remote_actor = None
ray.init(address="auto")

actors = [Actor.remote() for _ in range(100)]
for a in actors:
    other_ip = ray.get(a.id.remote())
    if other_ip != this_ip:
        remote_actor = a
        print("Found remote actor", other_ip, this_ip)
        break
else:
    assert False, "Didn't find remote actor"

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

    print("Size:", n)
    trials = []
    for _ in range(4):
        start = time.time()
        i = 0
        prev = data
        while time.time() - start < 1:
            prev = remote_actor.ping.remote(prev)
            ray.get(prev)
            i += 1
        trial = 1000 * (time.time() - start) / i
        print("Mean RTT", trial)
        trials.append(trial)
    print("Average:", sum(trials[1:])/(len(trials)-1))
    print()
