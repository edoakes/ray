import io
import os
import random
import time
import uuid
from typing import Dict

import boto3
from botocore.config import Config
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import ray
import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from util import parse_s3_uri

NUM_READ_ACTORS = 30
READ_ACTOR_CONCURRENCY = 4
NUM_PREPROCESS_ACTORS = 8
PREPROCESS_ACTOR_CONCURRENCY = 4
NUM_INFERENCE_ACTORS = 1
INFERENCE_ACTOR_CONCURRENCY = 2
NUM_WRITE_ACTORS = 2
WRITE_ACTOR_CONCURRENCY = 2

BATCH_SIZE = 100

INPUT_PATH = "s3://anonymous@ray-example-data/imagenet/metadata_file.parquet"
OUTPUT_PATH = "s3://anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/eoakes-resnet-core"
INPUT_LIMIT = 803_580


@ray.remote(num_cpus=0, max_concurrency=READ_ACTOR_CONCURRENCY)
class ReadActor:
    def __init__(self):
        self._s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2")

    def read(self, image_urls: list) -> list:
        results = []
        for image_url in image_urls:
            bucket, key = parse_s3_uri(image_url)
            with self._s3.open_input_stream(f"{bucket}/{key}") as f:
                results.append({"image_url": image_url, "bytes": f.read_buffer()})
        return results


@ray.remote(max_concurrency=PREPROCESS_ACTOR_CONCURRENCY)
class PreprocessActor:
    def __init__(self):
        weights = ResNet18_Weights.DEFAULT
        self._transform = transforms.Compose(
            [transforms.ToTensor(), weights.transforms()]
        )

    def preprocess(self, inputs: list) -> list:
        results = []
        for inp in inputs:
            image = Image.open(io.BytesIO(inp["bytes"])).convert("RGB")
            tensor = self._transform(image).numpy()
            results.append({"image_url": inp["image_url"], "tensor": tensor})
        return results


@ray.remote(num_gpus=1, max_concurrency=INFERENCE_ACTOR_CONCURRENCY)
class InferenceActor:
    def __init__(self):
        self._weights = ResNet18_Weights.DEFAULT
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = resnet18(weights=self._weights).to(self._device)
        self._model.eval()

    def predict(self, inputs: list) -> list:
        batch_tensor = torch.from_numpy(
            np.stack([inp["tensor"] for inp in inputs])
        ).to(self._device)
        with torch.inference_mode():
            predictions = self._model(batch_tensor)
            predicted_classes = predictions.argmax(dim=1).detach().cpu().tolist()
            predicted_labels = [
                self._weights.meta["categories"][c] for c in predicted_classes
            ]
        return [
            {"image_url": inp["image_url"], "label": label}
            for inp, label in zip(inputs, predicted_labels)
        ]


@ray.remote(max_concurrency=WRITE_ACTOR_CONCURRENCY)
class WriteActor:
    def __init__(self, bucket: str, output_dir: str):
        self._bucket = bucket
        self._output_dir = output_dir
        self._s3 = boto3.client("s3", config=Config(max_pool_connections=64))

    def write(self, inputs: list) -> str:
        table = pa.table({
            "image_url": [inp["image_url"] for inp in inputs],
            "label": [inp["label"] for inp in inputs],
        })
        buf = io.BytesIO()
        pq.write_table(table, buf)
        output_key = os.path.join(self._output_dir, f"{uuid.uuid4().hex}.parquet")
        self._s3.put_object(
            Bucket=self._bucket,
            Key=output_key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        return f"s3://{self._bucket}/{output_key}"


ray.init()
start_time_s = time.time()

# Read the input parquet to get image URLs.
s3_fs = pafs.S3FileSystem(anonymous=True, region="us-west-2")
input_table = pq.read_table(
    "ray-example-data/imagenet/metadata_file.parquet", filesystem=s3_fs
)
image_urls = input_table.column("image_url").to_pylist()[:INPUT_LIMIT]

output_bucket, output_dir = parse_s3_uri(OUTPUT_PATH)

read_actors = [ReadActor.remote() for _ in range(NUM_READ_ACTORS)]
preprocess_actors = [PreprocessActor.remote() for _ in range(NUM_PREPROCESS_ACTORS)]
inference_actors = [InferenceActor.remote() for _ in range(NUM_INFERENCE_ACTORS)]
write_actors = [
    WriteActor.remote(output_bucket, output_dir) for _ in range(NUM_WRITE_ACTORS)
]

inputs = [
    image_urls[i:i + BATCH_SIZE] for i in range(0, len(image_urls), BATCH_SIZE)
]
num_inputs = len(inputs)
pending_reads = set()
done_reads = set()
pending_preprocesses = set()
done_preprocesses = set()
pending_inferences = set()
done_inferences = set()
pending_writes = set()
done_writes = set()

print("Total num inputs:", num_inputs)

last_print_time_s = None
while True:
    time.sleep(0.001)

    all_remaining = (
        len(inputs)
        + len(pending_reads)
        + len(done_reads)
        + len(pending_preprocesses)
        + len(done_preprocesses)
        + len(pending_inferences)
        + len(done_inferences)
        + len(pending_writes)
    )
    if all_remaining == 0:
        break

    if not last_print_time_s or time.time() - last_print_time_s > 1:
        last_print_time_s = time.time()
        print("inputs remaining:", len(inputs))
        print("pending_reads:", len(pending_reads))
        print("done_reads:", len(done_reads))
        print("pending_preprocesses:", len(pending_preprocesses))
        print("done_preprocesses:", len(done_preprocesses))
        print("pending_inferences:", len(pending_inferences))
        print("done_inferences:", len(done_inferences))
        print("pending_writes:", len(pending_writes))
        print("done_writes:", len(done_writes))

    for r in ray.wait(list(pending_reads), timeout=0, fetch_local=False)[0]:
        pending_reads.remove(r)
        done_reads.add(r)

    for r in ray.wait(list(pending_preprocesses), timeout=0, fetch_local=False)[0]:
        pending_preprocesses.remove(r)
        done_preprocesses.add(r)

    for r in ray.wait(list(pending_inferences), timeout=0, fetch_local=False)[0]:
        pending_inferences.remove(r)
        done_inferences.add(r)

    for r in ray.wait(list(pending_writes), timeout=0, fetch_local=False)[0]:
        pending_writes.remove(r)
        done_writes.add(r)

    # Submit reads.
    target_reads = 2 * NUM_READ_ACTORS * READ_ACTOR_CONCURRENCY
    reads_to_submit = target_reads - (len(pending_reads) + len(done_reads))
    for _ in range(reads_to_submit):
        if not inputs:
            break
        read_actor = random.choice(read_actors)
        pending_reads.add(read_actor.read.remote(inputs.pop(0)))

    # Submit preprocesses.
    target_preprocesses = 2 * NUM_PREPROCESS_ACTORS * PREPROCESS_ACTOR_CONCURRENCY
    preprocesses_to_submit = target_preprocesses - (
        len(pending_preprocesses) + len(done_preprocesses)
    )
    for _ in range(preprocesses_to_submit):
        if not done_reads:
            break
        preprocess_actor = random.choice(preprocess_actors)
        pending_preprocesses.add(preprocess_actor.preprocess.remote(done_reads.pop()))

    # Submit inferences.
    target_inferences = 2 * NUM_INFERENCE_ACTORS * INFERENCE_ACTOR_CONCURRENCY
    inferences_to_submit = target_inferences - (
        len(pending_inferences) + len(done_inferences)
    )
    for _ in range(inferences_to_submit):
        if not done_preprocesses:
            break
        inference_actor = random.choice(inference_actors)
        pending_inferences.add(inference_actor.predict.remote(done_preprocesses.pop()))

    # Submit writes.
    target_writes = 2 * NUM_WRITE_ACTORS * WRITE_ACTOR_CONCURRENCY
    writes_to_submit = target_writes - len(pending_writes)
    for _ in range(writes_to_submit):
        if not done_inferences:
            break
        write_actor = random.choice(write_actors)
        pending_writes.add(write_actor.write.remote(done_inferences.pop()))

ray.get(list(done_writes))
print(f"Finished in {time.time() - start_time_s:.2f}s")
print(f"Outputs written to {OUTPUT_PATH}")
