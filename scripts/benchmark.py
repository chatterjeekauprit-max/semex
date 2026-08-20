import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from drift_sense.generator import generate_dataset_pair
from drift_sense.matcher import locate_full

# generate once - we're timing the matcher, not the generator
pairs = [generate_dataset_pair(seed=s, noise=True) for s in range(1, 6)]

for _ in range(5):                      # warmup: CUDA context, cuFFT plans
    locate_full(pairs[0][0], pairs[0][1])
torch.cuda.synchronize()

times = []
for _ in range(20):
    for s, r, t in pairs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        locate_full(s, r)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

times = np.array(times)
print(f"  n        : {len(times)}")
print(f"  median   : {np.median(times):.2f} ms")
print(f"  mean     : {times.mean():.2f} ms")
print(f"  p95      : {np.percentile(times, 95):.2f} ms")
print(f"  min / max: {times.min():.2f} / {times.max():.2f} ms")