"""Render episode-0 init states of two suites side by side per task.
Usage: python compare_two_suites_init.py <suiteA> <suiteB> <outname>
Left = suiteA, right = suiteB; rows = task ids. Output under scratch_render/<outname>/.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

A, B, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
RES, CAM, SETTLE = 320, "agentview", 12
OUTDIR = f"/workspace/LIBERO/scratch_render/{OUT}"
os.makedirs(OUTDIR, exist_ok=True)
bd = benchmark.get_benchmark_dict()
sA, sB = bd[A](), bd[B]()

def band(img, rgb, top=26):
    img = img.copy(); img[:top] = (img[:top]*0.35 + np.array(rgb)*0.65).astype(np.uint8); return img

def shot(suite, tid):
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    states = suite.get_task_init_states(tid)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=[CAM], camera_heights=RES, camera_widths=RES)
    env.seed(0); env.reset(); env.set_init_state(states[0])
    d = np.zeros(7); d[-1] = -1.0
    for _ in range(SETTLE): env.step(d.tolist())
    base = env.env; base.sim.forward()
    img = base.sim.render(camera_name=CAM, width=RES, height=RES)[::-1]
    env.close(); return img

rows = []
for tid in range(10):
    left = band(shot(sA, tid), (30, 90, 160))    # blue = suiteA
    right = band(shot(sB, tid), (170, 70, 30))    # orange = suiteB
    rows.append(np.concatenate([left, right], 1))
    print(f"[t{tid}] done")
sheet = np.concatenate(rows, 0)
p = os.path.join(OUTDIR, f"{OUT}_grid.png")
imageio.imwrite(p, sheet)
print(f"left={A} (blue)  right={B} (orange)\ngrid -> {p} shape={sheet.shape}")
