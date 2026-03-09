import io
import os
import random
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import boto3
import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import torch
from botocore.config import Config
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from util import parse_s3_uri

import ray

BATCH_SIZE = 100

READ_ACTORS_PER_NODE = 12
READ_ACTOR_CONCURRENCY = 16
PREPROCESS_ACTORS_PER_NODE = 8
PREPROCESS_ACTOR_CONCURRENCY = 4
INFERENCE_ACTORS_PER_NODE = 1
INFERENCE_ACTOR_CONCURRENCY = 4
WRITE_ACTORS_PER_NODE = 1
WRITE_ACTOR_CONCURRENCY = 4

INPUT_PATH = "s3://anonymous@ray-example-data/imagenet/metadata_file.parquet"
OUTPUT_PATH = "s3://anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/eoakes-resnet-core"
INPUT_LIMIT = 800_000
# INPUT_LIMIT = 100_000


@ray.remote(max_concurrency=READ_ACTOR_CONCURRENCY)
class ReadActor:
    def __init__(self):
        self._s3 = pafs.S3FileSystem(anonymous=True, region="us-west-2")

    def _fetch_single(self, image_url: str):
        bucket, key = parse_s3_uri(image_url)
        with self._s3.open_input_stream(f"{bucket}/{key}") as f:
            return {"image_url": image_url, "bytes": f.read_buffer(), "error": None}

    def read(self, image_urls: list) -> list:
        return list(map(self._fetch_single, image_urls))


@ray.remote(max_concurrency=PREPROCESS_ACTOR_CONCURRENCY)
class PreprocessActor:
    def __init__(self):
        self._resize_size = 232
        self._crop_size = 224
        # ImageNet normalization constants, shaped for CHW broadcast.
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _preprocess_one(self, inp: dict) -> dict:
        # Decode JPEG.
        image = Image.open(pa.BufferReader(inp["bytes"])).convert("RGB")
        # Resize short edge to 232 (PIL C code, releases GIL).
        w, h = image.size
        if h < w:
            new_h = self._resize_size
            new_w = int(w * self._resize_size / h)
        else:
            new_w = self._resize_size
            new_h = int(h * self._resize_size / w)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        # Center crop to 224x224.
        left = (new_w - self._crop_size) // 2
        top = (new_h - self._crop_size) // 2
        image = image.crop((left, top, left + self._crop_size, top + self._crop_size))
        # HWC uint8 → CHW float32, normalize. All numpy, no torch overhead.
        arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) * (1.0 / 255.0)
        arr = (arr - self._mean) * (1.0 / self._std)
        return {"image_url": inp["image_url"], "tensor": arr}

    def preprocess(self, inputs: list) -> list:
        return list(self._executor.map(self._preprocess_one, inputs))


@ray.remote(num_gpus=1, max_concurrency=INFERENCE_ACTOR_CONCURRENCY)
class InferenceActor:
    def __init__(self):
        self._weights = ResNet18_Weights.DEFAULT
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = resnet18(weights=self._weights).to(self._device)
        self._model.eval()
        self._transfer_stream = torch.cuda.Stream()
        # One pinned buffer per concurrent slot to avoid races.
        self._pinned_bufs = [
            torch.empty((BATCH_SIZE, 3, 224, 224), dtype=torch.float32, pin_memory=True)
            for _ in range(INFERENCE_ACTOR_CONCURRENCY)
        ]
        self._buf_lock = threading.Lock()
        self._available_bufs = list(range(INFERENCE_ACTOR_CONCURRENCY))

    def _acquire_buf(self):
        with self._buf_lock:
            return self._available_bufs.pop()

    def _release_buf(self, idx):
        with self._buf_lock:
            self._available_bufs.append(idx)

    def predict(self, inputs: list) -> list:
        stacked = np.stack([inp["tensor"] for inp in inputs])
        n = stacked.shape[0]
        buf_idx = self._acquire_buf()
        pinned = self._pinned_bufs[buf_idx][:n]
        pinned.copy_(torch.from_numpy(stacked))
        with torch.cuda.stream(self._transfer_stream):
            batch_tensor = pinned.to(self._device, non_blocking=True)
        # Record event so we know when the transfer (and thus pinned buf read) is done.
        transfer_done = self._transfer_stream.record_event()

        with torch.inference_mode():
            # Make the default stream wait for the transfer to finish on the GPU side.
            torch.cuda.current_stream().wait_event(transfer_done)
            predictions = self._model(batch_tensor)
            predicted_classes = predictions.argmax(dim=1).detach().cpu().tolist()
            predicted_labels = [
                self._weights.meta["categories"][c] for c in predicted_classes
            ]
        # Safe to release pinned buffer now — transfer is long done.
        self._release_buf(buf_idx)
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
        table = pa.table(
            {
                "image_url": [inp["image_url"] for inp in inputs],
                "label": [inp["label"] for inp in inputs],
            }
        )
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

# Shuffle image_urls to avoid throttling on specific prefixes.
# Probably could further optimize by sharding the prefixes among readers.
random.shuffle(image_urls)

output_bucket, output_dir = parse_s3_uri(OUTPUT_PATH)

# Get alive GPU node IDs and place actors evenly across them.
node_ids = [
    n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0
]
print(f"Found {len(node_ids)} GPU nodes")

actor_to_node = {}
read_actors = []
preprocess_actors = []
inference_actors = []
write_actors = []
read_actors_by_node = defaultdict(list)
preprocess_actors_by_node = defaultdict(list)
inference_actors_by_node = defaultdict(list)
write_actors_by_node = defaultdict(list)

for node_id in node_ids:
    label = {"ray.io/node-id": node_id}
    for _ in range(READ_ACTORS_PER_NODE):
        a = ReadActor.options(label_selector=label).remote()
        read_actors.append(a)
        read_actors_by_node[node_id].append(a)
        actor_to_node[a] = node_id
    for _ in range(PREPROCESS_ACTORS_PER_NODE):
        a = PreprocessActor.options(label_selector=label).remote()
        preprocess_actors.append(a)
        preprocess_actors_by_node[node_id].append(a)
        actor_to_node[a] = node_id
    for _ in range(INFERENCE_ACTORS_PER_NODE):
        a = InferenceActor.options(label_selector=label).remote()
        inference_actors.append(a)
        inference_actors_by_node[node_id].append(a)
        actor_to_node[a] = node_id
    for _ in range(WRITE_ACTORS_PER_NODE):
        a = WriteActor.options(label_selector=label).remote(output_bucket, output_dir)
        write_actors.append(a)
        write_actors_by_node[node_id].append(a)
        actor_to_node[a] = node_id

ref_to_node = {}  # ObjectRef -> node_id
ref_to_actor = {}  # ObjectRef -> actor handle
actor_pending = defaultdict(int)  # actor handle -> in-flight count

MAX_PER_READ = 2 * READ_ACTOR_CONCURRENCY
MAX_PER_PREPROCESS = 2 * PREPROCESS_ACTOR_CONCURRENCY
MAX_PER_INFERENCE = 2 * INFERENCE_ACTOR_CONCURRENCY
MAX_PER_WRITE = 2 * WRITE_ACTOR_CONCURRENCY

# Backpressure: limit each stage's (pending + done) to 2x downstream capacity.
num_nodes = len(node_ids)
MAX_READ_OUTPUTS = 2 * num_nodes * PREPROCESS_ACTORS_PER_NODE * MAX_PER_PREPROCESS
MAX_PREPROCESS_OUTPUTS = 2 * num_nodes * INFERENCE_ACTORS_PER_NODE * MAX_PER_INFERENCE
MAX_INFERENCE_OUTPUTS = 2 * num_nodes * WRITE_ACTORS_PER_NODE * MAX_PER_WRITE

inputs = [image_urls[i : i + BATCH_SIZE] for i in range(0, len(image_urls), BATCH_SIZE)]
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
        print("==================== PROGRESS ====================")
        print("inputs remaining:", len(inputs))
        print("pending_reads:", len(pending_reads))
        print("done_reads:", len(done_reads))
        print("pending_preprocesses:", len(pending_preprocesses))
        print("done_preprocesses:", len(done_preprocesses))
        print("pending_inferences:", len(pending_inferences))
        print("done_inferences:", len(done_inferences))
        print("pending_writes:", len(pending_writes))
        print("done_writes:", len(done_writes))
        print("==================================================")

    for r in ray.wait(list(pending_reads), timeout=0, fetch_local=False)[0]:
        pending_reads.remove(r)
        actor_pending[ref_to_actor.pop(r)] -= 1
        done_reads.add(r)

    for r in ray.wait(list(pending_preprocesses), timeout=0, fetch_local=False)[0]:
        pending_preprocesses.remove(r)
        actor_pending[ref_to_actor.pop(r)] -= 1
        done_preprocesses.add(r)

    for r in ray.wait(list(pending_inferences), timeout=0, fetch_local=False)[0]:
        pending_inferences.remove(r)
        actor_pending[ref_to_actor.pop(r)] -= 1
        done_inferences.add(r)

    for r in ray.wait(list(pending_writes), timeout=0, fetch_local=False)[0]:
        pending_writes.remove(r)
        actor_pending[ref_to_actor.pop(r)] -= 1
        done_writes.add(r)

    # Submit reads.
    while inputs and (len(pending_reads) + len(done_reads)) < MAX_READ_OUTPUTS:
        actor = min(read_actors, key=lambda a: actor_pending[a])
        if actor_pending[actor] >= MAX_PER_READ:
            break
        ref = actor.read.remote(inputs.pop(0))
        ref_to_actor[ref] = actor
        actor_pending[actor] += 1
        ref_to_node[ref] = actor_to_node[actor]
        pending_reads.add(ref)

    # Submit preprocesses.
    retry = []
    while (
        done_reads
        and (len(pending_preprocesses) + len(done_preprocesses))
        < MAX_PREPROCESS_OUTPUTS
    ):
        in_ref = done_reads.pop()
        node = ref_to_node.get(in_ref)
        local = preprocess_actors_by_node.get(node) or preprocess_actors
        actor = min(local, key=lambda a: actor_pending[a])
        if actor_pending[actor] >= MAX_PER_PREPROCESS:
            retry.append(in_ref)
            continue
        ref_to_node.pop(in_ref)
        ref = actor.preprocess.remote(in_ref)
        ref_to_actor[ref] = actor
        actor_pending[actor] += 1
        ref_to_node[ref] = actor_to_node[actor]
        pending_preprocesses.add(ref)
    done_reads.update(retry)

    # Submit inferences.
    retry = []
    while (
        done_preprocesses
        and (len(pending_inferences) + len(done_inferences)) < MAX_INFERENCE_OUTPUTS
    ):
        in_ref = done_preprocesses.pop()
        node = ref_to_node.get(in_ref)
        local = inference_actors_by_node.get(node) or inference_actors
        actor = min(local, key=lambda a: actor_pending[a])
        if actor_pending[actor] >= MAX_PER_INFERENCE:
            retry.append(in_ref)
            continue
        ref_to_node.pop(in_ref)
        ref = actor.predict.remote(in_ref)
        ref_to_actor[ref] = actor
        actor_pending[actor] += 1
        ref_to_node[ref] = actor_to_node[actor]
        pending_inferences.add(ref)
    done_preprocesses.update(retry)

    # Submit writes.
    retry = []
    while done_inferences:
        in_ref = done_inferences.pop()
        node = ref_to_node.get(in_ref)
        local = write_actors_by_node.get(node) or write_actors
        actor = min(local, key=lambda a: actor_pending[a])
        if actor_pending[actor] >= MAX_PER_WRITE:
            retry.append(in_ref)
            continue
        ref_to_node.pop(in_ref)
        ref = actor.write.remote(in_ref)
        ref_to_actor[ref] = actor
        actor_pending[actor] += 1
        ref_to_node[ref] = actor_to_node[actor]
        pending_writes.add(ref)
    done_inferences.update(retry)

ray.get(list(done_writes))
print(f"Finished in: {time.time() - start_time_s:.2f}s")
