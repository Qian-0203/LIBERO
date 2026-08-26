"""Render a contact sheet from a suite's EXISTING (already-generated) init states.
Read-only: never writes/overwrites any .pruned_init file, only PNGs under scratch_render/.
Usage: python render_suite_contact_sheet.py <suite_name>
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = sys.argv[1]
RES, CAM = 256, "agentview"
# Matches run_libero_eval.py's cfg.num_steps_wait=10 dummy-action settle before the real eval
# ever takes an observation -- without this, objects render mid-fall from their sampled init
# height (e.g. bowls appear to float) instead of resting on the surface the eval actually sees.
NUM_STEPS_WAIT = 10
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
RENDER_DIR = f"/workspace/LIBERO/scratch_render/{SUITE}"
os.makedirs(RENDER_DIR, exist_ok=True)

suite = benchmark.get_benchmark_dict()[SUITE]()
init_dir = os.path.join(get_libero_path("init_states"), SUITE)

tiles = []
for tid in range(suite.n_tasks):
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    states = torch.load(os.path.join(init_dir, task.init_states_file), weights_only=False)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=[CAM], camera_heights=RES, camera_widths=RES)
    env.seed(0); env.reset()
    env.set_init_state(states[0])
    for _ in range(NUM_STEPS_WAIT):
        env.step(DUMMY_ACTION)
    base = env.env; base.sim.forward()
    img = base.sim.render(camera_name=CAM, width=RES, height=RES)[::-1]
    env.close()
    img = img.copy(); img[:16] = (img[:16] * 0.25).astype(img.dtype)
    imageio.imwrite(os.path.join(RENDER_DIR, f"t{tid}_init.png"), img)
    tiles.append(img)
    print(f"[t{tid}] {task.name} rendered")

h, w, c = tiles[0].shape
sheet = np.full((2 * h, 5 * w, c), 255, tiles[0].dtype)
for i, t in enumerate(tiles):
    r, cc = divmod(i, 5)
    sheet[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = t
out = os.path.join(RENDER_DIR, f"{SUITE}_init_grid.png")
imageio.imwrite(out, sheet)
print(f"contact sheet -> {out}")
