import time
import zmq
import random
import json
from sentence_transformers import SentenceTransformer
import sys

model = SentenceTransformer(sys.argv[1]).half().cuda()
model.encode("test")

batching_window = 1 #ms

context_receiver = zmq.Context()
receiver = context_receiver.socket(zmq.PULL)
receiver.setsockopt(zmq.LINGER, 0)
receiver.bind("tcp://*:5555")

#Send embeddings back
context_sender = zmq.Context()
sender = context_sender.socket(zmq.PUB)
sender.setsockopt(zmq.LINGER, 0)
sender.bind("tcp://*:6666")

print("start listining")
while True:
    #  Wait for next request from client

    messages = []
    while receiver.poll(batching_window) == zmq.POLLIN:
        messages.append(receiver.recv_multipart())

    if len(messages) == 0:
        continue

    print("Received request:", len(messages), time.time() )
    # Compute embeddings
    texts = []
    sender_ids = []
    for sender_id, msg in messages:
        sender_ids.append(sender_id)
        texts.append(msg.decode('utf-8'))
      
    #Compute embeddings
    embeddings = model.encode(texts, convert_to_tensor=True, batch_size=256).cpu()

    # Send back the embeddings
    for idx in range(len(embeddings)):
        sender_id = sender_ids[idx]
        emb = embeddings[idx]
        #print("Respond", [sender_id, emb[0:3]])
        sender.send_multipart([sender_id, json.dumps(emb.tolist()).encode()])


 