from my_pkg import my_func

ray.init(runtime_env={"pip": ["requests"]})

@ray.remote
def task():
    import tensorflow
    tensorflow.do_something()


def main():
    ray.get(task.remote())


if __name__ == "__main__":
    main()
