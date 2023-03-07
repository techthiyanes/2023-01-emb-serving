import asyncio
from dataclasses import dataclass
import json
from typing import List

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from async_batcher import Batcher
import sentence_transformers

# --------------Configs and dependencies--------------

@dataclass
class Config:
	port: int = 8080
	max_batch_size: int = 256

config = Config(...)




model = sentence_transformers.SentenceTransformer('LaBSE').half().cuda()



# --------------Batcher Setup--------------

def translate(texts: List[str]):
	print("Len texts:", len(texts))
	emb = model.encode(texts, convert_to_numpy=True)
	return emb


batcher = Batcher(
	batch_prediction_fn=translate, 
	max_batch_size=config.max_batch_size
)


# --------------FastAPI Setup--------------

app = FastAPI()

origins = [
	f"http://localhost:{config.port}"
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/embed", status_code=201)
async def embed(query: str):
	emb = await batcher.predict(query)
	return JSONResponse({"emb": emb.tolist()})


@app.get("/")
async def root():
	return Response(
		content=json.dumps({"Status": "Alive"}), 
		status_code=status.HTTP_200_OK
	)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
	return JSONResponse({"Error": str(exc)})


@app.exception_handler(Exception)
async def final_exception_handler(request, err):
    base_error_message = f"Failed to execute: {request.method}: {request.url}"
    return JSONResponse(status_code=500, content={"message": f"{base_error_message}", "Details": str(err)})


# --------------Starting batcher--------------

@app.on_event("startup")
async def startup_event():
	loop = asyncio.get_running_loop()
	asyncio.create_task(batcher.start(loop))



if __name__ == "__main__":
	uvicorn.run("fastapi_example:app", host="0.0.0.0", port=config.port, reload=True)
