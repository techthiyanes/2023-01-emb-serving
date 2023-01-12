from sentence_transformers import SentenceTransformer
import time

model = SentenceTransformer("/model").cuda().half()


num_text = 1_000_000
input_text = ["testing the multilingual on AWS"] * num_text

start_time = time.time()
emb = model.encode(input_text, show_progress_bar=True, batch_size=1024, convert_to_tensor=True)
time_diff = time.time() - start_time

print(f"{num_text / time_diff} queries/second")
