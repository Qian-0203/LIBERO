"""Generate init states for a LIBERO suite and render a contact sheet.
Usage: python gen_suite_init_states.py <suite_name>
Runs inside the eval docker so the flattened MuJoCo state matches run_libero_eval.py.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = sys.argv[1]
N, RES, CAM = 50, 256, "agentview"
RENDER_DIR = f"/workspace/LIBERO/scratch_render/{SUITE}"

suite = benchmark.get_benchmark_dict()[SUITE]()
init_dir = os.path.join(get_libero_path("init_states"), SUITE)
os.makedirs(init_dir, exist_ok=True); os.makedirs(RENDER_DIR, exist_ok=True)

tiles = []
for tid in range(suite.n_tasks):
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=[CAM], camera_heights=RES, camera_widths=RES)
    env.seed(0)
    states, first = [], None
    for i in range(N):
        env.reset()
        states.append(np.asarray(env.get_sim_state(), dtype=np.float64))
        if i == 0:
            base = env.env; base.sim.forward()
            first = base.sim.render(camera_name=CAM, width=RES, height=RES)[::-1]
    env.close()
    arr = np.stack(states, 0)
    torch.save(arr, os.path.join(init_dir, task.init_states_file))
    print(f"[t{tid}] {task.name} shape={arr.shape} maxdelta={float(np.abs(arr-arr[0]).max()):.4f}")
    img = first.copy(); img[:16] = (img[:16]*0.25).astype(img.dtype)
    imageio.imwrite(os.path.join(RENDER_DIR, f"t{tid}_init.png"), img); tiles.append(img)

h, w, c = tiles[0].shape
sheet = np.full((2*h, 5*w, c), 255, tiles[0].dtype)
for i, t in enumerate(tiles):
    r, cc = divmod(i, 5); sheet[r*h:(r+1)*h, cc*w:(cc+1)*w] = t
imageio.imwrite(os.path.join(RENDER_DIR, f"{SUITE}_init_grid.png"), sheet)
print(f"contact sheet -> {RENDER_DIR}/{SUITE}_init_grid.png")
