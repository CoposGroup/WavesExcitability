#!/usr/bin/env python3
"""
simulate.py  --  Run RD model for a single Df, save stats to CSV.

USAGE (inside job):  python simulate.py --df 0.9 --seed 42 --outdir results
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
#from pde_utils import correlated_gaussian_field, laplacian
from my_tracker_module import track_wave          # ← import your function!
import argparse, numpy as np, pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

# --- fixed model params -------------------------------------------------
k0,k1,k2 = 0.00625,0.3125,1
k3,k4,k5 = 0.0625,0.05625,0.0625
k6,k7,k8 = 0.02083,0.001875,0.14062
k9,k10   = 0.25,0.025
Drt,Drd  = 0.08,0.4
sigma,s  = 0.75,4
f, alpha,beta = 10,1,1
size,dt,t_total,frame_int = 100,0.1,1000.0,0.1
save_every=int(frame_int/dt); dW_update=int(f/dt); n_steps=int(t_total/dt)

# ───────────── local replacements for pde_utils ───────────── #
def laplacian(Z, DX=1.0):
    """2-D five-point Laplacian with periodic BCs (grid spacing DX)."""
    return (np.roll(Z,  1, axis=1) + np.roll(Z, -1, axis=1) +
            np.roll(Z,  1, axis=0) + np.roll(Z, -1, axis=0) -
            4*Z) / DX**2

def double_derivative(Z, DX=1.0):
    """1-D second derivative with periodic BCs."""
    return (np.roll(Z, 1) + np.roll(Z, -1) - 2*Z) / DX**2

def correlated_gaussian_field(SIGMA, S, SHAPE, MEAN=1.0):
    """Spatially correlated Gaussian noise (periodic, std≈SIGMA)."""
    noise  = np.random.normal(0, 1, SHAPE)
    field  = gaussian_filter(noise, sigma=S, mode='wrap')
    return MEAN + SIGMA * field / np.std(field)
# ───────────────────────────────────────────────────────────── #

######### Wave Tracking STuff###############
def to_uint8(stack: np.ndarray) -> np.ndarray:
    vmin, vmax = stack.min(), stack.max()
    return (255*(stack - vmin)/(vmax - vmin + 1e-12)).astype(np.uint8)

def save_mp4(frames, path: str, fps=10):
    h, w = frames[0].shape[:2]
    vw = cv.VideoWriter(path,
                        cv.VideoWriter_fourcc(*'mp4v'),
                        fps, (w, h), True)
    for f in frames:
        vw.write(f)
    vw.release()
def find_wave_fronts(gray: np.ndarray, k: int = 10, min_dist: int = 10):
    """
    Pick up to `k` high-contrast points spread at least `min_dist` pixels apart.
    """
    # gradient magnitude as "image" for corner detection
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    gmag = cv.magnitude(gx, gy)                    # float32

    # goodFeaturesToTrack requires 8-bit or float; gmag already float
    corners = cv.goodFeaturesToTrack(gmag,
                                     maxCorners=k,
                                     qualityLevel=0.01,
                                     minDistance=min_dist,
                                     blockSize=3,
                                     useHarrisDetector=False)
    return corners.astype(np.float32)   # shape (≤k,1,2)

def track_wave(stack: np.ndarray,
               step: int      = 5,
               win_size: int  = 100,
               seeds: int     = 10,
               out_prefix: str = "wave"):
    """
    Track wave-front points and return hop velocities (px/frame).

    * No early-stop logic: runs through the entire stack.
    * “Dead” points are dropped forever once LK fails to find them.
    """
    STEP = max(1, step)
    u8   = to_uint8(stack)                                   # (T,H,W) → uint8
    bgr  = [cv.cvtColor(fr, cv.COLOR_GRAY2BGR) for fr in u8]

    # ─── Lucas–Kanade parameters ───
    lk = dict(winSize=(win_size, win_size),
              maxLevel=1,
              criteria=(cv.TERM_CRITERIA_EPS |
                        cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # ─── initial seed points ───
    old_gray = u8[0]
    p0 = find_wave_fronts(old_gray, k=seeds)                 # (N,1,2) float32
    active_keys = list(range(len(p0)))                       # 0…N-1

    # ─── drawing setup ───
    mask   = np.zeros_like(bgr[0])
    colour = np.random.randint(0, 255, (len(p0), 3))
    frames = []

    # track history per point: idx → list[(frame#, (x,y))]
    hist = {idx: [(0, tuple(pt[0]))] for idx, pt in enumerate(p0)}

    # ─── main loop over frames ───
    for f in range(1, len(u8)):
        new_gray = u8[f]
        p1, st, _ = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk)
        if p1 is None:
            break

        st = st.reshape(-1)
        good_idx = np.where(st == 1)[0]                      # survivors
        good_new = p1[st == 1].reshape(-1, 2)                # (M,2)

        frame = bgr[f].copy()
        for local_i, (x, y) in enumerate(good_new):
            idx = active_keys[good_idx[local_i]]             # global index
            x_prev, y_prev = hist[idx][-1][1]
            hist[idx].append((f, (x, y)))

            mask = cv.line(mask, (int(x_prev), int(y_prev)),
                                 (int(x),     int(y)),
                                 colour[idx].tolist(), 2)
            frame = cv.circle(frame, (int(x), int(y)),
                              4, colour[idx].tolist(), -1)

        frames.append(cv.add(frame, mask))
        old_gray = new_gray.copy()

        # ----- permanently remove lost points -----
        p0 = good_new.reshape(-1, 1, 2)
        active_keys = [active_keys[i] for i in good_idx]

        if len(p0) == 0:         # everything died → break early
            break

    # ─── compute velocities every STEP frames ───
    velocities = []
    for h in hist.values():
        for j in range(STEP, len(h)):
            fn_now,(x_now,y_now)   = h[j]
            fn_prev,(x_prev,y_prev)= h[j-STEP]
            if fn_now - fn_prev == STEP:
                velocities.append(np.hypot(x_now - x_prev,
                                           y_now - y_prev) / STEP)
    velocities = np.asarray(velocities)

    # ─── save overlay video ───
    vid_path = f"{out_prefix}_tracks.mp4"
    save_mp4([bgr[0]] + frames, vid_path)
    print(f"Overlay video saved  →  {vid_path}")

    return velocities
###########################################################


def reaction(A,B,C):
    return (k0+alpha*k1*A**3/(1+k2*A**2))*B-(k3+k4*(1+beta)*C)*A

def run_simulation(Df,seed=42):
    np.random.seed(seed)
    RT = 0.1+0.9*np.random.rand(size,size)
    RD = np.full((size,size),0.1)
    F  = np.zeros((size,size))
    dW = correlated_gaussian_field(sigma,s,(size,size),1.0)
    frames=[]

    for i in range(n_steps):
        R  = reaction(RT,RD,F)
        RT += dt*(R+Drt*laplacian(RT))
        RD += dt*(k5-k6*RD-R+Drd*laplacian(RD))
        F  += dt*(k7+k8*RT**2/(1+k9*RT**2)-k10*dW*F+Df*laplacian(F))
        if i % dW_update==0:
            dW=correlated_gaussian_field(sigma,s,(size,size),1.0)
        if i % save_every==0:
            frames.append(RT.copy())
    return np.stack(frames)

def main(df,seed,outdir):
    stack=run_simulation(df,seed)
    stack=stack[5000:]                # burn-in cut
    vel  = track_wave(stack,step=3,win_size=5,seeds=6,
                      out_prefix=f"Df{df:.2f}_{seed}")
    stats=dict(Df=df,
               mean=vel.mean(),
               median=np.median(vel),
               q25=np.percentile(vel,25),
               q75=np.percentile(vel,75),
               std=vel.std(),
               min=vel.min(),
               max=vel.max())
    pd.DataFrame([stats]).to_csv(Path(outdir)/f"Df{df:.3f}_stats.csv",
                                 index=False)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--df",type=float,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--outdir",default="results")
    args=ap.parse_args()
    Path(args.outdir).mkdir(exist_ok=True)
    main(args.df,args.seed,args.outdir)
