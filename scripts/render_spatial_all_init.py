"""Offscreen-render the init state (agentview) of all 10 LIBERO-Spatial tasks
and assemble a 2x5 contact sheet."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

RES = 384
CAM = "agentview"
OUTDIR = "/home/ec2-user/aliy/vla_ws/LIBERO/scratch_render/spatial_all"
os.makedirs(OUTDIR, exist_ok=True)


def label(img, text):
    """Crude top-left text bar by darkening a strip (no font dep)."""
    img = img.copy()
    img[:18, :, :] = (img[:18, :, :] * 0.25).astype(img.dtype)
    return img


def main():
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    n = suite.n_tasks
    tiles = []
    for tid in range(n):
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=[CAM],
                                 camera_heights=RES, camera_widths=RES)
        env.seed(0)
        env.reset()
        base = env.env
        base.sim.forward()
        img = base.sim.render(camera_name=CAM, width=RES, height=RES)[::-1]
        out = os.path.join(OUTDIR, f"spatial_t{tid}_{CAM}_init.png")
        imageio.imwrite(out, img)
        tiles.append(label(img, str(tid)))
        print(f"[t{tid}] {task.language}  ->  {out}")
        env.close()

    # 2 rows x 5 cols contact sheet
    cols, rows = 5, 2
    h, w, c = tiles[0].shape
    sheet = np.full((rows * h, cols * w, c), 255, dtype=tiles[0].dtype)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, cols)
        sheet[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = t
    sheet_path = os.path.join(OUTDIR, "spatial_all_init_grid.png")
    imageio.imwrite(sheet_path, sheet)
    print(f"\ncontact sheet -> {sheet_path}")


if __name__ == "__main__":
    main()
