import os
import random
import time
from typing import Dict

import boto3
from botocore.config import Config
import numpy as np
import ray

from util import list_s3_files, parse_s3_uri

NUM_READ_ACTORS = 2
READ_ACTOR_CONCURRENCY = 10
NUM_WRITE_ACTORS = 2
WRITE_ACTOR_CONCURRENCY = 10
NUM_MAP_ACTORS = 3
MAP_ACTOR_CONCURRENCY = 10
MAP_COMPUTE_TIME_S = 0.01
INPUT_URI = "s3://doggos-dataset/train"
OUTPUT_URI = "s3://anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/eoakes-proto"

@ray.remote(max_concurrency=READ_ACTOR_CONCURRENCY)
class ReadActor:
    def __init__(self, bucket: str):
        self._bucket = bucket
        self._s3 = boto3.client("s3", config=Config(max_pool_connections=64))

    def read(self, key: str) -> bytes:
        # TODO: this isn't using zero-copy serialization.
        r = self._s3.get_object(Bucket=self._bucket, Key=key)
        assert r["ResponseMetadata"]["HTTPStatusCode"] == 200
        return {"key": os.path.basename(key), "data": r["Body"].read()}

s3_client = None

@ray.remote
def read_task(bucket: str, key: str) -> bytes:
    global s3_client
    if s3_client is None:
        s3_client = boto3.client("s3", config=Config(max_pool_connections=64))

    r = s3_client.get_object(Bucket=bucket, Key=key)
    assert r["ResponseMetadata"]["HTTPStatusCode"] == 200
    return {"key": os.path.basename(key), "data": r["Body"].read()}


@ray.remote
class MapActor:
    def map(self, input: Dict) -> Dict:
        start_time_s = time.perf_counter()
        while time.perf_counter() - start_time_s < MAP_COMPUTE_TIME_S:
            pass

        return input


@ray.remote
def write_task(bucket: str, output_dir: str, input: Dict) -> bytes:
    global s3_client
    if s3_client is None:
        s3_client = boto3.client("s3", config=Config(max_pool_connections=64))

    output_path = os.path.join(output_dir, input["key"])
    r = s3_client.put_object(
        Bucket=bucket,
        Key=output_path,
        # TODO: this should be zero-copy.
        Body=input["data"],
        ContentType="application/octet-stream",
    )
    assert r["ResponseMetadata"]["HTTPStatusCode"] == 200
    return f"{bucket}/{input['key']}"


@ray.remote(max_concurrency=WRITE_ACTOR_CONCURRENCY)
class WriteActor:
    def __init__(self, bucket: str, output_dir: str):
        self._bucket = bucket
        self._output_dir = output_dir
        self._s3 = boto3.client("s3", config=Config(max_pool_connections=64))

    def write(self, input: Dict):
        output_path = os.path.join(self._output_dir, input["key"])
        r = self._s3.put_object(
            Bucket=self._bucket,
            Key=output_path,
            # TODO: this should be zero-copy.
            Body=input["data"],
            ContentType="application/octet-stream",
        )
        assert r["ResponseMetadata"]["HTTPStatusCode"] == 200
        return f"{self._bucket}/{input['key']}"

ray.init()
start_time_s = time.time()

input_bucket, input_dir = parse_s3_uri(INPUT_URI)
output_bucket, output_dir = parse_s3_uri(OUTPUT_URI)

read_actors = [ReadActor.remote(input_bucket) for _ in range(NUM_READ_ACTORS)]
map_actors = [MapActor.remote() for _ in range(NUM_MAP_ACTORS)]
write_actors = [WriteActor.remote(output_bucket, output_dir) for _ in range(NUM_WRITE_ACTORS)]

inputs = list(list_s3_files(input_bucket, input_dir))
num_inputs = len(inputs)
pending_reads = set()
done_reads = set()
pending_maps = set()
done_maps = set()
pending_writes = set()
done_writes = set()

print("Total num inputs:", num_inputs)

last_print_time_s = None
while True:
    time.sleep(0.001)

    if len(inputs) + len(pending_reads) + len(done_reads) + len(pending_maps) + len(done_maps) + len(pending_writes) == 0:
        break

    if not last_print_time_s or time.time() - last_print_time_s > 1:
        last_print_time_s = time.time()
        print("inputs remaining:", len(inputs))
        print("pending_reads:", len(pending_reads))
        print("done_reads:", len(done_reads))
        print("pending_maps:", len(pending_maps))
        print("done_maps:", len(done_maps))
        print("pending_writes:", len(pending_writes))
        print("done_writes:", len(done_writes))

    for r in ray.wait(list(pending_reads), timeout=0)[0]:
        pending_reads.remove(r)
        done_reads.add(r)

    for r in ray.wait(list(pending_maps), timeout=0)[0]:
        pending_maps.remove(r)
        done_maps.add(r)

    for r in ray.wait(list(pending_writes), timeout=0)[0]:
        pending_writes.remove(r)
        done_writes.add(r)

    target_reads = 2 * NUM_READ_ACTORS * READ_ACTOR_CONCURRENCY
    reads_to_submit = target_reads - (len(pending_reads) + len(done_reads))
    for _ in range(reads_to_submit):
        if len(inputs) == 0:
            break

        read_actor = random.choice(read_actors)
        pending_reads.add(read_actor.read.remote(inputs.pop(0)))
        # pending_reads.add(read_task.remote(input_bucket, inputs.pop(0)))

    target_maps = 2 * NUM_READ_ACTORS * READ_ACTOR_CONCURRENCY
    maps_to_submit = target_maps - (len(pending_maps) + len(done_maps))
    for _ in range(maps_to_submit):
        if len(done_reads) == 0:
            break

        map_actor = random.choice(map_actors)
        pending_maps.add(map_actor.map.remote(done_reads.pop()))
        # pending_maps.add(map_task.remote(input_bucket, inputs.pop(0)))

    target_writes = 2 * NUM_WRITE_ACTORS * WRITE_ACTOR_CONCURRENCY
    writes_to_submit = target_writes - (len(pending_writes))
    for _ in range(writes_to_submit):
        if len(done_maps) == 0:
            break

        write_actor = random.choice(write_actors)
        pending_writes.add(write_actor.write.remote(done_maps.pop()))
        # pending_writes.add(write_task.remote(output_bucket, output_dir, done_reads.pop()))

ray.get(list(done_writes))
assert len(done_writes) == num_inputs
print(f"Finished in {time.time() - start_time_s:.2f}s")
print(f"Outputs written to {OUTPUT_URI}")