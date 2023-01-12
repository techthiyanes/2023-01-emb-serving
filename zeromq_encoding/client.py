import zmq
import uuid
import time
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

#  Do 10 requests, waiting each time for a response
for idx in range(100):
    text = f"Hello {idx}"
    print("send", text, "identity:", identity)
    sender.send_multipart([identity, text.encode()])

    _, emb = receiver.recv_multipart()
    emb = json.loads(emb.decode())
    print("Response:", emb[0:3])


