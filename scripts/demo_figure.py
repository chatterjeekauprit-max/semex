
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, matplotlib.pyplot as plt, torch
from drift_sense.generator import generate_dataset_pair
from drift_sense.matcher import (degrade_reference, remove_periodic,
                                 prepare_template, locate_prepared,
                                 remove_periodic_gpu, zncc_gpu, DEV)

SEED, PATTERN, PERIOD = 1, "DRAM", 6
search, ref, truth = generate_dataset_pair(seed=SEED, noise=True, pattern=PATTERN)

tmpl_raw = degrade_reference(ref)
tmpl_res = remove_periodic(tmpl_raw, PERIOD)
prep = prepare_template(ref, period=PERIOD)
res = locate_prepared(search, prep, period=PERIOD)

s_gpu = torch.from_numpy(np.ascontiguousarray(search).copy()).to(DEV).float()
corr, vh, vw = zncc_gpu(remove_periodic_gpu(s_gpu, PERIOD), prep)
cmap_np = corr[:vh, :vw].cpu().numpy()

err = np.hypot(res["x"] - truth[0], res["y"] - truth[1])
fig, ax = plt.subplots(2, 3, figsize=(16, 10))

ax[0,0].imshow(ref, cmap="gray")
ax[0,0].set_title("1. Reference\n1000x1000 @ 1 nm/px (100x optic)")

ax[0,1].imshow(tmpl_raw, cmap="gray")
ax[0,1].set_title("2. Degraded template\nPSF blur + decimate -> 100x100 @ 10 nm/px")

ax[0,2].imshow(tmpl_res, cmap="gray")
ax[0,2].set_title("3. Lattice suppressed\naperiodic residual only")

ax[1,0].imshow(search, cmap="gray")
ax[1,0].add_patch(plt.Rectangle((truth[0]-50, truth[1]-50), 100, 100,
                                ec="lime", fc="none", lw=1.5))
ax[1,0].set_title("4. Search image\n1000x1000 @ 10 nm/px (10x optic)\ngreen = ground truth")

im = ax[1,1].imshow(cmap_np, cmap="inferno")
ax[1,1].plot(res["x"]-49.5, res["y"]-49.5, "c+", ms=14, mew=2)
ax[1,1].set_title(f"5. ZNCC correlation surface\nPSR = {res['psr']:.1f}")
plt.colorbar(im, ax=ax[1,1], fraction=0.046)

ax[1,2].imshow(search, cmap="gray")
ax[1,2].add_patch(plt.Rectangle((truth[0]-50, truth[1]-50), 100, 100,
                                ec="lime", fc="none", lw=2))
ax[1,2].add_patch(plt.Rectangle((res["x"]-50, res["y"]-50), 100, 100,
                                ec="red", fc="none", lw=1, ls="--"))
ax[1,2].set_xlim(truth[0]-160, truth[0]+160); ax[1,2].set_ylim(truth[1]+160, truth[1]-160)
ax[1,2].set_title(f"6. Result (zoomed)\ntruth ({truth[0]:.1f},{truth[1]:.1f})  "
                  f"pred ({res['x']:.1f},{res['y']:.1f})\nerror = {err:.3f} px = {err*10:.2f} nm")

for a in ax.ravel(): a.axis("off")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/demo_pipeline.png", dpi=140, bbox_inches="tight")
print("saved results/demo_pipeline.png")
plt.show()