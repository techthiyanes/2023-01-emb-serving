
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sentence_transformers
import logging
import time
import torch

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available()


model = sentence_transformers.SentenceTransformer('/model').half().cuda()


@app.get("/embed")
async def search(query: str):
    start_time = time.time()
    emb = model.encode(query, convert_to_tensor=False, convert_to_numpy=False, show_progress_bar=False).cpu().tolist()
    time_taken = time.time()-start_time
    return {
        "time": time_taken, 
        "emb": emb
    }
