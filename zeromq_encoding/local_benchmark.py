from sentence_transformers import SentenceTransformer
import time
import sys

num_queries = 10_000
model = SentenceTransformer(sys.argv[1]).half()
model.encode("test")
queries = ["testing the multilingual on AWS"]*num_queries

start_time = time.time()
emb = model.encode(queries, convert_to_tensor=True, batch_size=512)
time_diff = time.time() - start_time

print(f"{num_queries/time_diff:.2f} qps")