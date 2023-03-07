
import argparse
import time
import numpy as np
import tritonclient.http
import tqdm
import sys
from sentence_transformers import SentenceTransformer
import numpy as np

text = sys.argv[1]
batch_size = int(sys.argv[2])

model_name = f"python_onnx"
url = "127.0.0.1:8000"
model_version = "1"


triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

query = tritonclient.http.InferInput(name="TEXT", shape=(batch_size, 1), datatype="BYTES")
model_score = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)

#in_data = np.asarray([text] * batch_size, dtype=object)
in_data = np.asarray([[text]] * batch_size, dtype=object)
print(in_data)
query.set_data_from_numpy(in_data)

start_time = time.time()
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
)
end_time = time.time()
print(f"Call took {(end_time-start_time)*1000} ms")

emb = response.as_numpy("output")[0]
print(emb.shape, emb[0:3])

#Check embedding
#exit()
print("Check embedding")
model = SentenceTransformer("sentence-transformers/LaBSE", device="cpu")
emb_check = model.encode(text, convert_to_numpy=True)
diff = np.max(np.abs(emb - emb_check))
print("Diff:", diff)