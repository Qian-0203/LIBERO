"""Render the actual eval init states of libero_spatial (2 bowls) vs
libero_spatial_3bowl (3 bowls), side by side per task.

For each task we restore episode-0 of the saved .pruned_init for each suite
(the exact state the eval feeds to set_init_state), settle a few steps, and
render agentview. Output: a 10-row x 2-col sheet (left = 2-bowl, right = 3-bowl).
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

RES = 320
CAM = "agentview"
OUTDIR = "/workspace/LIBERO/scratch_render/compare_2v3bowl"
SETTLE = 12  # dummy zero-action steps so bowls rest before the shot
os.makedirs(OUTDIR, exist_ok=True)

bd = benchmark.get_benchmark_dict()
suites = {"2bowl": bd["libero_spatial"](), "3bowl": bd["libero_spatial_3bowl"]()}


def band(img, top, rgb):
    img = img.copy()
    strip = img[:top].astype(np.float32)
    img[:top] = (strip * 0.35 + np.array(rgb) * 0.65).astype(np.uint8)
    return img


def shot(suite, tid):
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    states = suite.get_task_init_states(tid)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=[CAM], camera_heights=RES, camera_widths=RES)
    env.seed(0)
    env.reset()
    env.set_init_state(states[0])
    dummy = np.zeros(7); dummy[-1] = -1.0  # open gripper, no motion
    for _ in range(SETTLE):
        env.step(dummy.tolist())
    base = env.env
    base.sim.forward()
    img = base.sim.render(camera_name=CAM, width=RES, height=RES)[::-1]
    n = 0
    for b in ["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"]:
        try:
            base.get_object(b); n += 1
        except Exception:
            pass
    env.close()
    return img, n


rows = []
for tid in range(10):
    left, n2 = shot(suites["2bowl"], tid)
    right, n3 = shot(suites["3bowl"], tid)
    left = band(left, 26, (30, 90, 160))    # blue strip = 2 bowls
    right = band(right, 26, (170, 70, 30))  # orange strip = 3 bowls
    row = np.concatenate([left, right], axis=1)
    rows.append(row)
    imageio.imwrite(os.path.join(OUTDIR, f"t{tid}_2v3.png"), row)
    print(f"[t{tid}] 2bowl n={n2}  3bowl n={n3}  {suites['3bowl'].get_task(tid).name}")

sheet = np.concatenate(rows, axis=0)
out = os.path.join(OUTDIR, "compare_2v3bowl_grid.png")
imageio.imwrite(out, sheet)
print(f"\nleft column = 2-bowl (blue strip), right column = 3-bowl (orange strip)")
print(f"grid -> {out}  shape={sheet.shape}")
