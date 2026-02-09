from typing import Dict, Iterator, Tuple
from urllib.parse import urlparse

import boto3
import pyarrow.parquet as pq


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    path = parsed.path.lstrip("/")
    return bucket, path


def list_s3_files(bucket: str, prefix: str) -> Iterator[str]:
    """
    List all object keys under an S3 prefix.

    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (e.g. 'images/train/')

    Yields:
        Full S3 object keys
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def read_urls_from_parquet(path: str) -> Dict:
    table = pq.read_table(
        "s3://anonymous@ray-example-data/imagenet/metadata_file.parquet"
    )
    urls = []
    for chunk in table:
        urls.extend(str(url) for url in chunk)

    return urls
