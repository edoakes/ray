import time

import ray
from ray.data.datasource import RowBasedFileDatasink

MAP_COMPUTE_TIME_S = 0.01
INPUT_URI = "s3://doggos-dataset/train"
OUTPUT_URI = "s3://anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/eoakes-baseline"

ray.init()
start_time_s = time.time()


class BinaryFileDatasink(RowBasedFileDatasink):
    def __init__(self, path, *, column="bytes", file_format="png", **kwargs):
        super().__init__(path, file_format=file_format, **kwargs)
        self.column = column

    def write_row_to_file(self, row, file):
        file.write(row[self.column])

class Map:
    def __call__(self, input):
        start_time_s = time.perf_counter()
        while time.perf_counter() - start_time_s < MAP_COMPUTE_TIME_S:
            pass

        return input

ray.data.read_binary_files(INPUT_URI).map(Map, concurrency=3).write_datasink(BinaryFileDatasink(OUTPUT_URI))
print(f"Finished in {time.time()-start_time_s:.2f}s")
print(f"Outputs written to {OUTPUT_URI}")