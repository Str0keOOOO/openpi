"""Reserve GPU 4 and 7 memory to approximately 90% total usage.
Run: CUDA_VISIBLE_DEVICES=4,7 python scripts/reserve_gpus_4_7.py
Keep this process alive to retain the allocations; Ctrl-C releases them.
"""
import signal
import time
import torch

TARGET = 0.90
CHUNK_MIB = 256
allocations = []

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise SystemExit("Expected CUDA_VISIBLE_DEVICES=4,7 and exactly 2 visible GPUs")

for device in range(2):
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    target_bytes = int(total * TARGET)
    need = max(0, target_bytes - used)
    held = []
    chunk = CHUNK_MIB * 1024 * 1024
    try:
        while need > 0:
            size = min(chunk, need)
            held.append(torch.empty(size, dtype=torch.uint8, device=f"cuda:{device}"))
            need -= size
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"GPU {device}: OOM before target; retained {sum(x.numel() for x in held)/total:.1%} of total")
    allocations.append(held)
    free_now, _ = torch.cuda.mem_get_info(device)
    print(f"Physical GPU {('4','7')[device]}: total={total/2**30:.2f} GiB, "
          f"currently used={(total-free_now)/total:.1%}", flush=True)

print("Holding allocations. Press Ctrl-C to release.", flush=True)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    pass
