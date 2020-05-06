import ray
from ray import serve

import joblib
import requests
import s3fs

class SKLearnBackend:
    def __init__(self):
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open('ray-serve-blog/unigram_vectorizer.joblib', 'rb') as f:
            self.vectorizer = joblib.load(f)
        with fs.open('ray-serve-blog/unigram_tf_idf_transformer.joblib', 'rb') as f:
            self.preprocessor = joblib.load(f)
        with fs.open('ray-serve-blog/unigram_tf_idf_classifier.joblib', 'rb') as f:
            self.classifier = joblib.load(f)

    @serve.accept_batch
    def __call__(self, requests):
        texts = [request.data for request in requests]
        vectorized = self.vectorizer.transform(texts)
        transformed = self.preprocessor.transform(vectorized)
        return self.classifier.predict(transformed).astype(int).tolist()

serve.init(blocking=True)

# First, we create it as a backend.
config = {"max_batch_size": 100}
serve.create_backend("sklearn_backend", SKLearnBackend, config=config)

# Let's hook it up to an HTTP endpoint.
serve.create_endpoint("sentiment_endpoint", route="/sentiment")
serve.set_traffic("sentiment_endpoint", {"sklearn_backend": 1.0})

# Test that HTTP is working.
result = requests.get("http://127.0.0.1:8000/sentiment", data="I love RayServe!").text
print("result:", result)
