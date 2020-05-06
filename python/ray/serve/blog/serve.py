import ray
from ray import serve

import joblib
import requests
import s3fs

TEST_CASE = "RayServe eases the pain of model serving"

# Avoid some annoying warnings from sklearn being printed to the terminal.
ray.init(log_to_driver=False)

# Start the RayServe cluster.
serve.init(blocking=True)

class SKLearnBackend:
    def __init__(self):
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open('ray-serve-blog/unigram_vectorizer.joblib', 'rb') as f:
            self.vectorizer = joblib.load(f)
        with fs.open('ray-serve-blog/unigram_tf_idf_transformer.joblib',
                     'rb') as f:
            self.preprocessor = joblib.load(f)
        with fs.open('ray-serve-blog/unigram_tf_idf_classifier.joblib',
                     'rb') as f:
            self.classifier = joblib.load(f)

    @serve.accept_batch
    def __call__(self, requests):
        texts = [request.data for request in requests]
        vectorized = self.vectorizer.transform(texts)
        transformed = self.preprocessor.transform(vectorized)
        results = []
        for result in self.classifier.predict(transformed):
            if result == 1:
                results.append("POSITIVE")
            else:
                results.append("NEGATIVE")
        return results


# First, we create a simple sklearn backend.
config = {"max_batch_size": 100}
serve.create_backend("sklearn_backend", SKLearnBackend, config=config)

# Let's hook it up to an HTTP endpoint.
serve.create_endpoint("sentiment_endpoint", route="/sentiment")
serve.set_traffic("sentiment_endpoint", {"sklearn_backend": 1.0})

# Test that HTTP is working.
result = requests.get(
    "http://127.0.0.1:8000/sentiment", data=TEST_CASE).text
print("Result for '{}': {}".format(TEST_CASE, result))

# Uh, oh. That should be positive! Let's try upgrading to a pre-trained
# PyTorch model using transformers.

from transformers import pipeline

class PyTorchBackend:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")

    @serve.accept_batch
    def __call__(self, requests):
        texts = [str(request.data) for request in requests]
        return [result["label"] for result in self.classifier(texts)]

serve.create_backend("pytorch_backend", PyTorchBackend, config=config)
serve.set_traffic("sentiment_endpoint", {"pytorch_backend": 1.0})
result = requests.get(
    "http://127.0.0.1:8000/sentiment", data=TEST_CASE).text
print("Result for '{}': {}".format(TEST_CASE, result))

# Yay! It's better now and RayServe made it easy to switch.
