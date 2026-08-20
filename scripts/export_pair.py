"""Export one generated (search, reference) pair as PNG files plus ground truth."""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from PIL import Image
from drift_sense.generator import generate_dataset_pair

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--pattern", default="DRAM", choices=["DRAM", "FINFET"])
ap.add_argument("--outdir", default="generated_data")
args = ap.parse_args()

os.makedirs(args.outdir, exist_ok=True)
search, ref, truth = generate_dataset_pair(seed=args.seed, noise=True,
                                           pattern=args.pattern)

sp = os.path.join(args.outdir, f"search_{args.pattern}_{args.seed}.png")
rp = os.path.join(args.outdir, f"reference_{args.pattern}_{args.seed}.png")
Image.fromarray(search).save(sp)
Image.fromarray(ref).save(rp)

print(f"  search    : {sp}   {search.shape}  10 nm/px")
print(f"  reference : {rp}   {ref.shape}   1 nm/px")
print(f"  GROUND TRUTH centre (x, y) = ({truth[0]:.2f}, {truth[1]:.2f})")