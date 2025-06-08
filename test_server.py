import time
import argparse
import requests
import numpy as np
from PIL import Image
from io import BytesIO

# Replace with your actual /run endpoint
RUN_URL = "https://api.cortex.cerebrium.ai/v4/p-328695ed/mtailor-assessment/run"
# Optional healthcheck endpoint (common pattern)
HEALTH_URL = RUN_URL.replace("/run", "/health")

def preprocess_image_url(url):
    resp = requests.get(url)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr.tolist()

def call_run(image_url):
    payload = {"input_image": image_url}
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLTMyODY5NWVkIiwiaWF0IjoxNzQ5Mzc5NTY3LCJleHAiOjIwNjQ5NTU1Njd9.Gq2YpD5KBX823O4YVo-7-eMbpdMKk-Q8yTqvE2iEYlVe_A_khe-j4PnWw0F4d8PCMSySBMV5THiae9LETdNRu-HhQHM-fbyHFEXzZoJ_HIpssjMOBlb6eAQqTpDVJupRISUD_k-Z-W3kfd1r21NxCG3oJY2Mdn26EhjXBwVO1Ycm1Shum2-cWeKGR7pe003C57rZGorog1TCr8w-OEaas7wcjgngu4ax3bKszhojQ3LzFuAXHpY3iJFaQjO8aQOpDdOEugZstCecEr0Nfjm5vB-3RO8k2ptW6bdEouJbyjueQWczFFI4F3caP40W_s4mTiVxH6--eU-PgHui-YWX8A',
        'Content-Type': 'application/json'
    }
    start = time.time()
    resp = requests.post(RUN_URL, json=payload, headers=headers)
    elapsed = (time.time() - start) * 1000
    if resp.ok:
        data = resp.json()
        print(data)
    else:
        print(f"Error {resp.status_code}: {resp.text}")


def run_concurrency_test(image_url, num_requests=10):
    from threading import Thread
    results = []
    def worker():
        resp = requests.post(RUN_URL, json={"input_image": image_url})
        results.append(resp.status_code)
    threads = [Thread(target=worker) for _ in range(num_requests)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    dur = time.time() - t0
    print(f"🕒 Concurrency test: {num_requests} calls in {dur:.2f}s -> statuses: {set(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="URL of an image to send for prediction")
    parser.add_argument("--run-tests", action="store_true",
                        help="Run deployment tests: health, sample inference, concurrency")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of concurrent requests in concurrency test")
    args = parser.parse_args()

    if args.run_tests:
        print(" Deployment tests starting...")
        sample_url = "https://th.bing.com/th/id/OIP.Q6cBixBci6LZJuyEmHxubwHaE7?rs=1&pid=ImgDetMain"
        print("Sample inference test:")
        call_run(sample_url)
        print("Concurrency test:")
        run_concurrency_test(sample_url, num_requests=args.concurrency)
    elif args.image:
        print(f"Sending image URL: {args.image}")
        call_run(args.image)
    else:
        print(" Usage: Provide --image <URL> or --run-tests")

