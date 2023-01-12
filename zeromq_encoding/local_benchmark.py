from sentence_transformers import SentenceTransformer
import time


num_queries = 1_000
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
queries = ["test this is a test"]*num_queries

start_time = time.time()
emb = model.encode(queries, convert_to_tensor=True)
time_diff = time.time() - start_time

print(f"{num_queries/time_diff:.2f} qps")