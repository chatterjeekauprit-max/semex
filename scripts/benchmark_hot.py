import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from drift_sense.generator import generate_dataset_pair
from drift_sense.matcher import prepare_template, locate_prepared

pairs = [generate_dataset_pair(seed=s, noise=True) for s in range(1, 6)]
prepped = [(s, prepare_template(r), t) for s, r, t in pairs]

for _ in range(10):
    locate_prepared(prepped[0][0], prepped[0][1])
torch.cuda.synchronize()

times = []
for _ in range(20):
    for s, tm, t in prepped:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        locate_prepared(s, tm)
        torch.cuda.synchronize(); times.append((time.perf_counter()-t0)*1000)

times = np.array(times)
print(f"  median: {np.median(times):.2f} ms   p95: {np.percentile(times,95):.2f} ms   "
      f"min: {times.min():.2f} ms")

# accuracy must be unchanged
errs = [np.hypot(locate_prepared(s,tm)["x"]-t[0], locate_prepared(s,tm)["y"]-t[1])
        for s, tm, t in prepped]
print(f"  errors: {[f'{e:.2f}' for e in errs]}")