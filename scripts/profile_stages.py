import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from drift_sense.generator import generate_dataset_pair
from drift_sense.matcher import (degrade_reference, remove_periodic,
                                 zncc_map_normalized, top_k_peaks,
                                 peak_to_sidelobe, subpixel_peak)

s, r, t = generate_dataset_pair(seed=1, noise=True)

def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

tm = degrade_reference(r)
tmr = remove_periodic(tm)
sr = remove_periodic(s.astype(np.float32))
corr = zncc_map_normalized(sr, tmr)

print(f"  degrade_reference   : {timeit(lambda: degrade_reference(r)):7.2f} ms")
print(f"  remove_periodic(ref): {timeit(lambda: remove_periodic(tm)):7.2f} ms")
print(f"  remove_periodic(srch): {timeit(lambda: remove_periodic(s.astype(np.float32))):7.2f} ms")
print(f"  zncc_map_normalized : {timeit(lambda: zncc_map_normalized(sr, tmr)):7.2f} ms")
print(f"  top_k_peaks         : {timeit(lambda: top_k_peaks(corr, 901, 901, 5)):7.2f} ms")
print(f"  peak_to_sidelobe    : {timeit(lambda: peak_to_sidelobe(corr, 901, 901, 400, 400)):7.2f} ms")