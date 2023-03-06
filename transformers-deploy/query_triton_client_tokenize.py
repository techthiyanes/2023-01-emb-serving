
import argparse
import time
import numpy as np
import tritonclient.http
import tqdm
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer

text = sys.argv[1]


model_name = f"transformer_onnx_model"
url = "127.0.0.1:8000"
model_version = "1"


triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
enc = tokenizer(text, return_tensors="np")
nb_tokens = enc['input_ids'].shape
print(enc)
print("nb_tokens", nb_tokens)
print("dtype", enc['input_ids'].dtype)


input_ids = tritonclient.http.InferInput("input_ids", nb_tokens, "INT32")
input_ids.set_data_from_numpy(enc['input_ids'].astype(np.int32), binary_data=False)

attention_mask = tritonclient.http.InferInput("attention_mask", nb_tokens, "INT32")
attention_mask.set_data_from_numpy(enc['attention_mask'].astype(np.int32), binary_data=False)

token_type_ids = tritonclient.http.InferInput("token_type_ids", nb_tokens, "INT32")
token_type_ids.set_data_from_numpy(enc['token_type_ids'].astype(np.int32), binary_data=False)

model_score = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)



start_time = time.time()
response = triton_client.infer(
    model_name=model_name, model_version=model_version, inputs=[input_ids, attention_mask, token_type_ids], outputs=[model_score]
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