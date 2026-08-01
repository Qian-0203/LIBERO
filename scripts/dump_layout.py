"""Dump episode-0 xy of landmarks + bowls for each task of a suite, to plan
hard-negative placement. Usage: python dump_layout.py <suite>"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial_3bowl"
NAMES = ["akita_black_bowl_1","akita_black_bowl_2","akita_black_bowl_3",
         "plate_1","glazed_rim_porcelain_ramekin_1","cookies_1","flat_stove_1","wooden_cabinet_1"]
suite = benchmark.get_benchmark_dict()[SUITE]()
init_dir = os.path.join(get_libero_path("init_states"), SUITE)
import torch
for tid in range(suite.n_tasks):
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    states = torch.load(os.path.join(init_dir, task.init_states_file))
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0); env.reset(); env.set_init_state(states[0]); base = env.env
    print(f"\n[t{tid}] {task.name}")
    for n in NAMES:
        try:
            q = base.sim.data.get_joint_qpos(base.get_object(n).joints[-1])
            print(f"    {n:34s} x={q[0]:+.3f} y={q[1]:+.3f} z={q[2]:+.3f}")
        except Exception:
            # fixtures may not have a free joint; try body xpos
            try:
                bid = base.sim.model.body_name2id(base.get_object(n).root_body)
                p = base.sim.data.body_xpos[bid]; print(f"    {n:34s} x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f} (body)")
            except Exception as e:
                print(f"    {n:34s} <n/a>")
    env.close()
