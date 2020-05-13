import argparse
import json
import os
import time

import ray

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arg-size", type=str, required=True, help="'small' or 'large'")

@ray.remote
def generate_object(use_small):
    if use_small:
        return b"ok"
    else:
        return np.zeros(1024 * 1024, dtype=np.uint8)


# Worker nodes only have 4 CPUs, force spread.
@ray.remote(num_cpus=4)
class Actor2:
    def __init__(self, other, use_small):
        self.other = other
        self.my_object = generate_object.remote(use_small)
        ray.get(self.my_object)

    def ping(self, arg):
        self.other.pong.remote(self.my_object)


# Worker nodes only have 4 CPUs, force spread.
@ray.remote(num_cpus=4)
class Actor1:
    def __init__(self, obj_size):
        self.rtts = []
        self.my_object = generate_object.remote(obj_size)
        ray.get(self.my_object)

    def do_ping_pong(self, other):
        self.start_time = time.time()
        other.ping.remote(self.my_object)

    def pong(self, arg):
        self.rtts.append(time.time() - self.start_time)

    def get_rtts(self):
        return self.rtts


def do_ray_init(args):
    internal_config = {"record_ref_creation_sites": 0}
    if os.environ.get("BY_VAL_ONLY", False):
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


def main(use_small):
    do_ray_init(use_small)

    actor1 = Actor1.remote(use_small)
    actor2 = Actor2.remote(actor1, use_small)
    trials = 1100
    for i in range(trials):
        print("iter {}/{}".format(i + 1, trials))
        time.sleep(0.01)
        actor1.do_ping_pong.remote(actor2)

    latencies = [rtt / 2 for rtt in ray.get(actor1.get_rtts.remote())[100:]]
    print("avg:", sum(latencies) / len(latencies))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.arg_size == "small":
        use_small = True
    else:
        use_small = False
    main(use_small)
