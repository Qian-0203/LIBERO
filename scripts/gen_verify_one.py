import os, sys
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
SUITE, TID = sys.argv[1], int(sys.argv[2])
BOWLS=["akita_black_bowl_1","akita_black_bowl_2","akita_black_bowl_3"]
suite=benchmark.get_benchmark_dict()[SUITE](); task=suite.get_task(TID)
bddl=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file)
init_dir=os.path.join(get_libero_path("init_states"),SUITE)
env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=128,camera_widths=128); env.seed(0)
states=[]
for i in range(50):
    env.reset(); states.append(np.asarray(env.get_sim_state(),dtype=np.float64))
arr=np.stack(states,0); torch.save(arr, os.path.join(init_dir, task.init_states_file))
# verify separation over the saved states
ms=1e9
for s in arr:
    env.set_init_state(s); b=env.env
    xy={n:b.sim.data.get_joint_qpos(b.get_object(n).joints[-1])[:2] for n in BOWLS}
    for i in range(3):
        for j in range(i+1,3):
            ms=min(ms,float(np.linalg.norm(xy[BOWLS[i]]-xy[BOWLS[j]])))
env.close()
print(f"[t{TID}] regen+saved shape={arr.shape} min_sep={ms:.3f} {'OK' if ms>=0.12 else '!!OVERLAP'}")
