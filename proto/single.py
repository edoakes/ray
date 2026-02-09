from typing import Dict

import boto3
from botocore.config import Config
from util import list_s3_files, parse_s3_uri

import ray

INPUT_URI = "s3://doggos-dataset/train"
OUTPUT_URI = "s3://anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/eoakes-test/"


@ray.remote(max_concurrency=10)
class ReadActor:
    def __init__(self, bucket: str):
        self._bucket = bucket
        self._s3 = boto3.client("s3", config=Config(max_pool_connections=64))

    def read(self, key: str) -> bytes:
        r = self._s3.get_object(Bucket=self._bucket, Key=key)
        return {"key": key, "data": r["Body"].read()}


@ray.remote
class MapActor:
    def map(self, input: Dict) -> Dict:
        for _ in range(100000):
            pass

        return input


@ray.remote(max_concurrency=10)
class WriteActor:
    def __init__(self, bucket: str):
        self._bucket = bucket
        self._s3 = boto3.client("s3", config=Config(max_pool_connections=64))

    def write(self, input: Dict):
        self._s3.put_object(
            Bucket=self._bucket,
            Key=input["key"],
            Body=input["data"],
            ContentType="application/octet-stream",
        )
        return f"{self._bucket}/{input['key']}"


input_bucket, input_dir = parse_s3_uri(INPUT_URI)
output_bucket, output_dir = parse_s3_uri(OUTPUT_URI)

read_actor = ReadActor.remote(input_bucket)
write_actor = WriteActor.remote(output_bucket)

inputs = list(list_s3_files(input_bucket, input_dir))
num_inputs = len(inputs)
pending_reads = set()
done_reads = set()
pending_writes = set()
done_writes = set()

while True:
    import time

    time.sleep(0.01)

    if len(inputs) + len(pending_reads) + len(done_reads) + len(pending_writes) == 0:
        break

    print("inputs remaining:", len(inputs))
    print("pending_reads:", len(pending_reads))
    print("done_reads:", len(pending_reads))
    print("pending_writes:", len(pending_reads))
    print("done_writes:", len(pending_reads))

    for r in ray.wait(list(pending_reads), timeout=0)[0]:
        pending_reads.remove(r)
        done_reads.add(r)

    for r in ray.wait(list(pending_writes), timeout=0)[0]:
        pending_writes.remove(r)
        done_writes.add(r)

    reads_to_submit = 20 - (len(pending_reads) + len(done_reads))
    for _ in range(reads_to_submit):
        if len(inputs) == 0:
            break

        pending_reads.add(read_actor.read.remote(inputs.pop(0)))

    writes_to_submit = 20 - (len(pending_writes))
    for _ in range(writes_to_submit):
        if len(done_reads) == 0:
            break

        pending_writes.add(write_actor.write.remote(done_reads.pop()))

assert len(done_writes) == num_inputs
