"""
uvicorn client_fastapi:app --workers 1 --port 8111
ab -c 1 -n 100 http://127.0.0.1:8111/embed?query=test
"""
import zmq
import uuid
from fastapi import FastAPI
import json


identity = str(uuid.uuid4()).encode('ascii')

#  Sender queue
context_sender = zmq.Context()
sender = context_sender.socket(zmq.PUSH)
sender.setsockopt(zmq.LINGER, 0)
sender.connect("tcp://localhost:5555")

# Receiver queue
context_receiver = zmq.Context()
receiver = context_receiver.socket(zmq.SUB)
receiver.setsockopt(zmq.LINGER, 0)
receiver.setsockopt(zmq.SUBSCRIBE, identity)
receiver.connect("tcp://localhost:6666")




app = FastAPI()


@app.get("/embed")
async def embed(query):
    print("send", query, "identity:", identity)
    sender.send_multipart([identity, query.encode()])

    _, emb = receiver.recv_multipart()
    #emb = json.loads(emb.decode())
    #print("Response:", emb[0:3])
    return emb
    


