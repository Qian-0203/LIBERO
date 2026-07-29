"""Generate init states for the libero_spatial_3bowl suite AND render a contact
sheet to eyeball the three-bowl placement.

Run inside the eval docker (mujoco 2.3.2 / robosuite 1.4.1) so the flattened
MuJoCo state layout matches what run_libero_eval.py's set_init_state expects.

For each task: build the env from its BDDL, seed(0) to mirror get_libero_env,
reset N times capturing env.get_sim_state() (the same pre-settle flattened state
the eval restores per trial), and torch.save the (N, state_dim) array to
init_files/libero_spatial_3bowl/<task>.pruned_init.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = "libero_spatial_3bowl"
N = 50
RES = 256
CAM = "agentview"
RENDER_DIR = "/workspace/LIBERO/scratch_render/spatial_3bowl"


def main():
    suite = benchmark.get_benchmark_dict()[SUITE]()
    init_root = get_libero_path("init_states")
    init_dir = os.path.join(init_root, SUITE)
    os.makedirs(init_dir, exist_ok=True)
    os.makedirs(RENDER_DIR, exist_ok=True)

    tiles = []
    for tid in range(suite.n_tasks):
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_names=[CAM], camera_heights=RES, camera_widths=RES
        )
        env.seed(0)

        states = []
        first_img = None
        for i in range(N):
            env.reset()
            states.append(np.asarray(env.get_sim_state(), dtype=np.float64))
            if i == 0:
                base = env.env
                base.sim.forward()
                first_img = base.sim.render(camera_name=CAM, width=RES, height=RES)[::-1]
                # report the three bowl xy so we can confirm they are distinct
                for name in ["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"]:
                    j = base.get_object(name).joints[-1]
                    q = base.sim.data.get_joint_qpos(j)
                    print(f"    {name:20s} xyz={np.round(q[:3], 3)}")
        env.close()

        arr = np.stack(states, axis=0)
        # sanity: states must vary across the N trials (not all identical)
        spread = float(np.abs(arr - arr[0]).max())
        out = os.path.join(init_dir, task.init_states_file)
        torch.save(arr, out)
        print(f"[t{tid}] {task.name}\n    shape={arr.shape} max|Δ vs trial0|={spread:.4f} -> {out}")

        img = first_img.copy()
        img[:16, :, :] = (img[:16, :, :] * 0.25).astype(img.dtype)
        imageio.imwrite(os.path.join(RENDER_DIR, f"t{tid}_init.png"), img)
        tiles.append(img)

    cols, rows = 5, 2
    h, w, c = tiles[0].shape
    sheet = np.full((rows * h, cols * w, c), 255, dtype=tiles[0].dtype)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, cols)
        sheet[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = t
    sheet_path = os.path.join(RENDER_DIR, "spatial_3bowl_grid.png")
    imageio.imwrite(sheet_path, sheet)
    print(f"\ncontact sheet -> {sheet_path}")


if __name__ == "__main__":
    main()
