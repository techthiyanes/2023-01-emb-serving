#  Copyright 2022, Lefebvre Dalloz Services
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse

import numpy as np
import tritonclient.http
import tqdm
import sys


model_name = f"transformer_onnx_inference"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

triton_query = tritonclient.http.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
model_score = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)


####
from fastapi import FastAPI
import json

app = FastAPI()
@app.get("/embed")
async def embed(query):
    triton_query.set_data_from_numpy(np.asarray([query] * batch_size, dtype=object))
    response = triton_client.infer(
        model_name=model_name, model_version=model_version, inputs=[triton_query], outputs=[model_score]
    )

    emb = response.as_numpy("output")[0]
    return emb.tolist()

