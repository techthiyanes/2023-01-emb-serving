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
import tritonclient.grpc.aio as grpcclient
import tqdm
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
import asyncio

text = sys.argv[1]

model_name = f"transformer_onnx_inference"
url = "127.0.0.1:8001"
model_version = "1"
batch_size = 1

async def main():
    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    assert triton_client.is_model_ready(
        model_name=model_name, model_version=model_version
    ), f"model {model_name} not yet ready"

    model_metadata = await triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
    model_config = await triton_client.get_model_config(model_name=model_name, model_version=model_version)

    query = grpcclient.InferInput(name="TEXT", shape=(batch_size,), datatype="BYTES")
    model_score = grpcclient.InferRequestedOutput(name="output") #, binary_data=False)

    query.set_data_from_numpy(np.asarray([text] * batch_size, dtype=object))
    response = await triton_client.infer(
        model_name=model_name, model_version=model_version, inputs=[query], outputs=[model_score]
    )

    emb = response.as_numpy("output")[0]
    print(emb.shape, emb[0:3])

    #Check embedding
    print("Check embedding")
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1", device="cpu")
    emb_check = model.encode(text, convert_to_numpy=True)
    diff = np.max(np.abs(emb - emb_check))
    print("Diff:", diff)

asyncio.run(main())