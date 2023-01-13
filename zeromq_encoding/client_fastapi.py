"""
uvicorn client_fastapi:app --port 8111 --workers 1 
ab -c 1 -n 100 http://127.0.0.1:8111/embed?query=testing+the+multilingual+on+AWS
ab -c 100 -n 10000 http://127.0.0.1:8112/embed?query=testing+the+multilingual+on+AWS
"""
import zmq
import uuid
from fastapi import FastAPI
import json
import numpy

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    sender_id = socket.recv(flags=flags)
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = msg #buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape']).tolist()

def recv_multipart(socket):
    _, emb = socket.recv_multipart()
    return emb.decode()

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
    #print("send", query, "identity:", identity)
    sender.send_multipart([identity, query.encode()])

    emb = recv_multipart(receiver)
    return emb
    


