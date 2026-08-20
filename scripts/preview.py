import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, matplotlib.pyplot as plt
from drift_sense.generator import generate_dataset_pair

search, ref, truth = generate_dataset_pair(seed=1, noise=True, pattern="FINFET")
x0, y0 = truth[0] - 49.5, truth[1] - 49.5

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(ref, cmap="gray"); ax[0].set_title("Reference 1000x1000 @ 1nm/px")
ax[1].imshow(search, cmap="gray"); ax[1].set_title("Search 1000x1000 @ 10nm/px")
ax[1].add_patch(plt.Rectangle((x0, y0), 100, 100, ec="lime", fc="none", lw=1.5))
ax[2].imshow(search[int(y0):int(y0)+100, int(x0):int(x0)+100], cmap="gray")
ax[2].set_title("Target region (100x100)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig("results/preview.png", dpi=150); plt.show()