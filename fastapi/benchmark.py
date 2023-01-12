import time
import requests

url = "http://localhost:2380/embed"
text = "What is the wifi password"

num_calls = 100
avg_time = 0
avg_encoding_time = 0

for _ in range(num_calls):
    start_time = time.time()
    response = requests.get(url, params={'query': text}).json()
    time_diff = time.time() - start_time
  
    encoding_time = response['time']

    avg_time += time_diff
    avg_encoding_time += encoding_time


avg_time /= num_calls
avg_encoding_time /= num_calls

print(f"Avg End2end time: {avg_time*1000:.0f} ms")
print(f"Avg Encoding time: {avg_encoding_time*1000:.0f} ms")
