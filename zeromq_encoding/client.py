import zmq
import uuid
import time
import json
import numpy 

def recv_array(socket, flags=0, copy=True, track=False):
    """recv a numpy array"""
    sender_id = socket.recv(flags=flags)
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = msg #buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

def recv_multipart(socket):
    _, emb = socket.recv_multipart()
    return json.loads(emb.decode())

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

#  Do 10 requests, waiting each time for a response
for idx in range(100):
    text = f"Hello {idx}"
    print("send", text, "identity:", identity)
    sender.send_multipart([identity, text.encode()])

    emb = recv_multipart(receiver)
    print("Response:", emb[0:3])


