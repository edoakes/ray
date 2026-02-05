import ray

@ray.remote
class ReadInputs:
    def __init__(self):
        pass


@ray.remote
class Infer:
    def __init__(self):
        pass


@ray.remote
class WriteOutputs:
    def __init__(self):
        pass
