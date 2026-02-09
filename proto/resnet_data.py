from __future__ import annotations

import io
import time
import uuid

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

import ray
from ray.data.expressions import download

BATCH_SIZE = 100
NUM_GPU_NODES = 8
INPUT_PATH = "s3://anonymous@ray-example-data/imagenet/metadata_file.parquet"
OUTPUT_PATH = f"s3://ray-data-write-benchmark/{uuid.uuid4().hex}"
OUTPUT_PATH = "s3://anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/eoakes-resnet-data"

weights = ResNet18_Weights.DEFAULT
transform = transforms.Compose([transforms.ToTensor(), weights.transforms()])


def deserialize_image(row):
    image = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
    # NOTE: Remove the `bytes` column since we don't need it anymore. This is done by
    # the system automatically on Ray Data 2.51+ with the `with_column` API.
    del row["bytes"]
    row["image"] = np.array(image)
    return row


def transform_image(row):
    row["norm_image"] = transform(row["image"]).numpy()
    # NOTE: Remove the `image` column since we don't need it anymore. This is done by
    # the system automatically on Ray Data 2.51+ with the `with_column` API.
    del row["image"]
    return row


class ResNetActor:
    def __init__(self):
        self.weights = weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(weights=self.weights).to(self.device)
        self.model.eval()

    def __call__(self, batch):
        torch_batch = torch.from_numpy(batch["norm_image"]).to(self.device)
        # NOTE: Remove the `norm_image` column since we don't need it anymore. This is
        # done by the system automatically on Ray Data 2.51+ with the `with_column`
        # API.
        del batch["norm_image"]
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [
                self.weights.meta["categories"][i] for i in predicted_classes
            ]
            batch["label"] = predicted_labels
            return batch


start_time = time.time()


ds = (
    ray.data.read_parquet(INPUT_PATH)
    # .limit(800_000)
    .limit(100_000)
    .with_column("bytes", download("image_url"))
    .map(fn=deserialize_image)
    .map(fn=transform_image)
    .map_batches(
        fn=ResNetActor,
        batch_size=BATCH_SIZE,
        num_gpus=1.0,
        concurrency=NUM_GPU_NODES,
    )
    .select_columns(["image_url", "label"])
)
ds.write_parquet(OUTPUT_PATH)


print(f"Finished in: {time.time() - start_time:.2f}s")
