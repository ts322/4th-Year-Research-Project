#!/usr/bin/env python3
"""
collect_params_valid.py
Sample (p1,p2,p3), validate by building a mesh via Docker OpenFOAM,
and save ONLY the valid parameters P to a compact NPZ.

Usage:
  python VAE/vae_data_prep.py \
    --out-root ~/ResearchProject/4th-Year-Research-Project/2D_Reactor/VAE/vae_data_prep.py \
    --n-samples 100 \
    --docker-image opencfd/openfoam-default:2506

Assumptions:
- Template case at ./swakless-1 with blockMeshDict containing
  $xmin/$xmax/$ymin/$ymax placeholders and room for polyLine edges.
"""

import os, math, uuid, shutil, argparse, subprocess
from datetime import datetime
from typing import Tuple, List
import numpy as np

# ---------- Geometry helpers (minimal, same shape logic you use) ----------
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.stats import qmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sin_line(x, b, a, c, off):
    return b + (a * np.sin(c * (x - off)))

def smooth_segment(x, y, n_add, k=5):
    x_p = [x[n_add-k], x[n_add], x[n_add+k]]
    y_mid_new = (((y[n_add-k] + y[n_add+k]) / 2) + y[n_add]) / 2
    y_p = [y[n_add-k], y_mid_new, y[n_add+k]]
    x_new = np.linspace(x[n_add-k], x[n_add+k], 2*k)
    y_new = interp1d([x[n_add-k], x[n_add], x[n_add+k]], y_p, kind="quadratic")(x_new)
    x[n_add-k:n_add+k] = x_new
    y[n_add-k:n_add+k] = y_new
    return x, y

def build_channel_curves(p1: float, p2: float, p3: float,
                         base_x=(0,3), n_points=180, n_add=50):
    if not (0.1 <= p1 <= 0.70): raise ValueError
    if not (3.0 <= p2 <= 6.0):  raise ValueError
    if not (0.0 <= p3 <= math.pi/2): raise ValueError

    x = np.linspace(base_x[0], base_x[1], n_points)
    y1 = sin_line(x, 0.5, p1, p2, p3)
    y2 = sin_line(x, 0.0, 0.25, p2, p3)

    x1, x2 = x.copy(), x.copy()
    addL = list(np.linspace(x1[0]-1.0, x1[0], n_add, endpoint=False))
    addR = list(np.flip(np.linspace(x1[-1]+1.0, x1[-1], n_add, endpoint=False)))

    fL = interpolate.interp1d([addL[0], addL[-1]], [y2[0]+0.5, y1[0]], kind='linear')
    fR = interpolate.interp1d([addR[0], addR[-1]], [y1[-1], y2[-1]+0.5], kind='linear')

    y1 = np.append(np.append(fL(addL), y1), fR(addR))
    y2 = np.append(np.append([y2[0]]*n_add, y2), [y2[-1]]*n_add)
    x1 = np.append(np.append(addL, x1), addR)
    x2 = np.append(np.append(addL, x2), addR)

    x1,y1 = smooth_segment(x1,y1,n_add);           x2,y2 = smooth_segment(x2,y2,n_add)
    x1,y1 = smooth_segment(x1,y1,n_add+n_points);  x2,y2 = smooth_segment(x2,y2,n_add+n_points)
    return x1, y1, x2, y2

def make_polyline_points(x, y, z):
    return [f"(\t{float(x[i])}\t{float(y[i])}\t{float(z)}\t)" for i in range(len(x))]

def render_blockMeshDict(template_lines: List[bytes], x1,y1,x2,y2) -> List[str]:
    lines = [str(l).split("b'")[-1].split("\\n")[0] for l in template_lines]
    s = []; l = 0; s.append(lines[l])
    while "polyLine" not in lines[l]:
        s.append(lines[l]); l += 1
    i = 0; c = 0
    while i < len(s):
        if '$' in s[i] and c in [0,4,7,3]:
            s[i]=(s[i].replace('$xmin',str(x2[0])).replace('$xmax',str(x1[0]))
                       .replace('$ymax',str(y1[0])).replace('$ymin',str(y2[0])))
            c += 1; i += 1
        elif '$' in s[i] and c in [2,6,5,1]:
            s[i]=(s[i].replace('$xmin',str(x1[-1])).replace('$xmax',str(x2[-1]))
                       .replace('$ymax',str(y1[-1])).replace('$ymin',str(y2[-1])))
            c += 1; i += 1
        else:
            i += 1
    nums = ["0 1","4 5","3 2","7 6"]
    l21 = make_polyline_points(x2,y2,0.0); l22 = make_polyline_points(x2,y2,0.1)
    l11 = make_polyline_points(x1,y1,0.0); l12 = make_polyline_points(x1,y1,0.1)
    for pair, poly in zip(nums, [l21,l22,l11,l12]):
        s.append(f"\tpolyLine {pair} ( {poly[0]}"); s += poly[1:]; s.append(")")
    s.append(");")
    while "boundary" not in lines[l]: l += 1
    s += lines[l:]
    return s

# ---------- Run OpenFOAM in Docker ----------
def docker_run(image: str, workdir_host: str, cmd: str):
    return subprocess.run(
        ["docker","run","--rm","-v",f"{workdir_host}:/home/openfoam","-w","/home/openfoam",
         image,"bash","-lc",cmd],
        capture_output=True, text=True, check=False
    )

def run_blockmesh_and_check(image: str, case_dir: str):
    _ = docker_run(image, case_dir, "blockMesh > log.blockMesh 2>&1")
    cm = docker_run(image, case_dir, "checkMesh > log.checkMesh 2>&1; cat log.checkMesh")
    log = cm.stdout

    # quick parse
    def first_int(label):
        for ln in log.splitlines():
            if label in ln:
                toks = ln.replace(":"," ").replace("="," ").split()
                for t in toks:
                    if t.isdigit(): return int(t)
        return None
    def last_float(label):
        for ln in log.splitlines():
            if label in ln:
                nums=[]
                for t in ln.replace(","," ").replace(":"," ").split():
                    try: nums.append(float(t))
                    except: pass
                if nums: return nums[-1]
        return None

    cells = first_int("cells:")
    maxNonOrtho = last_float("Max non-orthogonality")
    maxSkewness = last_float("Max skewness")
    maxAspectRatio = last_float("Max aspect ratio")

    valid = True
    if not cells or cells <= 0: valid = False
    if maxNonOrtho is not None and maxNonOrtho > 70: valid = False
    if maxSkewness is not None and maxSkewness > 4: valid = False
    if maxAspectRatio is not None and maxAspectRatio > 100: valid = False
    return valid, log

# ---------- Case writing ----------
def write_case(base_case_dir: str, out_dir: str, x1,y1,x2,y2):
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    shutil.copytree(base_case_dir, out_dir)
    with open(os.path.join(base_case_dir,"system","blockMeshDict"),"rb") as f:
        tmpl = f.readlines()
    bmd = render_blockMeshDict(tmpl, x1,y1,x2,y2)
    with open(os.path.join(out_dir,"system","blockMeshDict"),"w") as f:
        for line in bmd: f.write(line+"\n")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Collect ONLY valid (p1,p2,p3) for VAE.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--base-case-dir", default="./swakless-1")
    ap.add_argument("--docker-image", default="opencfd/openfoam-default:2506")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p1-min", type=float, default=0.1)
    ap.add_argument("--p1-max", type=float, default=0.5)
    ap.add_argument("--p2-min", type=float, default=3.0)
    ap.add_argument("--p2-max", type=float, default=6.0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    U = qmc.LatinHypercube(d=3).random(n=args.n_samples)
    lower = np.array([args.p1_min, args.p2_min, 0.0])
    upper = np.array([args.p1_max, args.p2_max, math.pi/2])
    samples = qmc.scale(U, lower, upper)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = os.path.join(os.path.expanduser(args.out_root), f"Pgen_{ts}")
    os.makedirs(root, exist_ok=True)
    base = os.path.abspath(args.base_case_dir)

    P_valid = []
    kept = 0

    for i,(p1,p2,p3) in enumerate(samples):
        uid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        case_dir = os.path.join(root, uid)
        print(f"[{i+1}/{len(samples)}] p=({p1:.3f},{p2:.3f},{p3:.3f}) -> {uid}")

        # geometry + case
        try:
            x1,y1,x2,y2 = build_channel_curves(p1,p2,p3)
        except ValueError:
            # out-of-bounds (shouldn't happen due to sampling), skip
            continue
        write_case(base, case_dir, x1,y1,x2,y2)

        # validate via Docker
        valid, log = run_blockmesh_and_check(args.docker_image, case_dir)
        with open(os.path.join(case_dir,"log.checkMesh"),"w") as fh:
            fh.write(log)

        if valid:
            P_valid.append([p1,p2,p3])
            kept += 1

    if not P_valid:
        raise RuntimeError("No valid meshes found. Adjust bounds or quality thresholds.")

    P = np.array(P_valid, dtype=np.float32)
    out_npz = os.path.join(root, "P_valid.npz")
    np.savez_compressed(out_npz, P=P)

    print(f"\nSaved ONLY valid parameters to: {out_npz}")
    print(f"P shape: {P.shape}")
    print(f"Cases root (each attempt stored): {root}")

if __name__ == "__main__":
    main()
