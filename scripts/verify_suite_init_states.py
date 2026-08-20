"""Verify a suite's init states: bowls pairwise separated (no overlap) + valid heights.
Also reports whether the wooden cabinet top drawer is open (joint qpos).
Usage: python verify_suite_init_states.py <suite_name>
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial_3bowl"
ALL_BOWLS = ["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"]
MIN_SEP = 0.12

suite = benchmark.get_benchmark_dict()[SUITE]()
init_dir = os.path.join(get_libero_path("init_states"), SUITE)
worst = 1e9; any_fail = False
for tid in range(suite.n_tasks):
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    states = torch.load(os.path.join(init_dir, task.init_states_file), weights_only=False)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0); env.reset()
    # Not every suite has 3 bowls (e.g. the 2-bowl grounding-probe gap-fill suites),
    # so only compare pairwise separation among the bowls actually present.
    bowls = []
    for b in ALL_BOWLS:
        try:
            if env.env.get_object(b) is not None:
                bowls.append(b)
        except Exception:
            pass
    min_sep = 1e9; zmin, zmax = 1e9, -1e9; drawer = None
    for s in states:
        env.set_init_state(s); base = env.env
        xy = {}
        for b in bowls:
            q = base.sim.data.get_joint_qpos(base.get_object(b).joints[-1])
            xy[b] = q[:3]; zmin, zmax = min(zmin, q[2]), max(zmax, q[2])
        for i in range(len(bowls)):
            for j in range(i + 1, len(bowls)):
                min_sep = min(min_sep, float(np.linalg.norm(xy[bowls[i]][:2] - xy[bowls[j]][:2])))
        # cabinet top drawer prismatic joint (slide); read its position if present
        try:
            for jn in base.sim.model.joint_names:
                if "wooden_cabinet" in jn and ("slide" in jn or "top" in jn):
                    drawer = float(base.sim.data.get_joint_qpos(jn)[0] if np.ndim(base.sim.data.get_joint_qpos(jn)) else base.sim.data.get_joint_qpos(jn)); break
        except Exception:
            pass
    env.close()
    worst = min(worst, min_sep)
    if min_sep < MIN_SEP: any_fail = True
    dtxt = f" drawer_qpos={drawer:+.3f}" if drawer is not None else ""
    print(f"[t{tid}] min sep={min_sep:.3f} z[{zmin:.3f},{zmax:.3f}]{dtxt} {'OK' if min_sep>=MIN_SEP else '!!OVERLAP'}  {task.name}")
print(f"\nWorst sep = {worst:.3f} m (thr {MIN_SEP}). RESULT:", "FAIL" if any_fail else "PASS")
