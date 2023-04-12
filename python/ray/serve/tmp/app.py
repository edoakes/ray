from ray import serve

@serve.deployment
class A:
    def __call__(self, *args):
        import time
        time.sleep(10)
        return "ok"

a = A.bind()
