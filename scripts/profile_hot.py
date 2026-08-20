import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from drift_sense.generator import generate_dataset_pair
from drift_sense.matcher import (prepare_template, remove_periodic_gpu,
                                 zncc_gpu, top_k_peaks_fast,
                                 peak_to_sidelobe, subpixel_peak, DEV)

s_np, r, t = generate_dataset_pair(seed=1, noise=True)
prep = prepare_template(r)

def timeit(fn, n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

s_gpu = torch.from_numpy(np.ascontiguousarray(s_np)).to(DEV).float()
s_res = remove_periodic_gpu(s_gpu)
corr, vh, vw = zncc_gpu(s_res, prep)

print(f"  upload+float    : {timeit(lambda: torch.from_numpy(np.ascontiguousarray(s_np)).to(DEV).float()):6.2f} ms")
print(f"  remove_periodic : {timeit(lambda: remove_periodic_gpu(s_gpu)):6.2f} ms")
print(f"  zncc_gpu        : {timeit(lambda: zncc_gpu(s_res, prep)):6.2f} ms")
print(f"  top_k_fast      : {timeit(lambda: top_k_peaks_fast(corr, vh, vw, 5)):6.2f} ms")
print(f"  peak_to_sidelobe: {timeit(lambda: peak_to_sidelobe(corr, vh, vw, 400, 400)):6.2f} ms")
print(f"  subpixel        : {timeit(lambda: subpixel_peak(corr, 400, 400)):6.2f} ms")