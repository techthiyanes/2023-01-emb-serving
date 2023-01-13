import time
import zmq
import random
import json
from transformers import AutoModel, AutoTokenizer
import sys
import torch
import time
from multiprocessing import Process
import multiprocessing as mp

def send_array(socket, sender_id, A, flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send(sender_id, flags|zmq.SNDMORE)
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def send_multipart(socket, sender_id, emb):
    socket.send_multipart([sender_id, json.dumps(emb.tolist()).encode()])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



encode_queue = mp.Queue(int(sys.argv[2]))
def emb_worker(model_name):
    device = "cuda"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.half().to(device)

    print("start emb worker")
    while True:
        # Compute embeddings
        sender_ids, texts = encode_queue.get()
 
        print("Received request:", len(texts), time.time() )

        #Compute embeddings
        start_time = time.time()
        with torch.inference_mode():
            encoded = tokenizer(texts, truncation=True, padding=True, return_tensors="pt", max_length=256).to(device)
            model_output = model(**encoded)
            embeddings = mean_pooling(model_output, encoded['attention_mask']).cpu().numpy()
        #print(f"Compute emb: {(time.time()-start_time)*1000:.2f}ms")

        # Send back the embeddings
        send_back_queue.put([sender_ids, embeddings])
     



#Send back process
send_back_queue = mp.Queue()
def send_back_worker():
    print("Start send_back_worker")
    
    #Send embeddings back
    context_sender = zmq.Context()
    sender = context_sender.socket(zmq.PUB)
    sender.setsockopt(zmq.LINGER, 0)
    sender.bind("tcp://*:6666")

    while True:
        #Queue get data
        sender_ids, embeddings = send_back_queue.get()
        for idx in range(len(embeddings)):
            send_multipart(sender, sender_ids[idx], embeddings[idx])


if __name__ == "__main__":
    model_name = sys.argv[1]
    workers = int(sys.argv[2])

    batching_window = 10
    context_receiver = zmq.Context()
    receiver = context_receiver.socket(zmq.PULL)
    receiver.setsockopt(zmq.LINGER, 0)
    receiver.bind("tcp://*:5555")
    

    #Send back worker
    p = Process(target=send_back_worker)
    p.start()

    #Emb worker
    for worker_id in range(workers):
        p = Process(target=emb_worker, args=[model_name])
        p.start()

 
    print("Start listing to zMQ")
    while True:
        messages = []
        while receiver.poll(batching_window) == zmq.POLLIN or encode_queue.full():
            messages.append(receiver.recv_multipart())

        if len(messages) == 0:
            continue

        print("Received request:", len(messages), time.time())

        # Compute embeddings
        texts = []
        sender_ids = []
        for sender_id, msg in messages:
            sender_ids.append(sender_id)
            texts.append(msg.decode('utf-8'))
        
        encode_queue.put([sender_ids, texts])