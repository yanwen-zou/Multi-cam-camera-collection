# prune_ckpt_remove_position_ids.py
import torch
from pathlib import Path
import sys

src_path = Path("/home/ryan/Documents/GitHub/AirExo-2-test/dependencies/ControlNet/models/control_sd15_ini.ckpt")
if not src_path.exists():
    print("ERROR: source file not found:", src_path); sys.exit(2)

dst_path = src_path.with_name(src_path.stem + "_pruned.ckpt")
bak_path = src_path.with_suffix(".ckpt.bak")

print("Backing up original to:", bak_path)
if not bak_path.exists():
    src_path.rename(bak_path)
else:
    print("Backup already exists:", bak_path)

print("Loading checkpoint (to CPU)...")
ckpt = torch.load(str(bak_path), map_location="cpu")

# 如果 ckpt 本身就是 state_dict（正好与你的输出一样），就直接操作
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    sd = ckpt["state_dict"]
    wrapped = True
else:
    sd = ckpt
    wrapped = False

keys_to_remove = [k for k in list(sd.keys()) if "position_ids" in k]
print("Found keys to remove:", keys_to_remove)

if not keys_to_remove:
    print("No position_ids keys found; exiting.")
    # 如果你想仍然保存一个拷贝（不删除），把下面两行取消注释
    # if wrapped:
    #     ckpt["state_dict"] = sd
    # torch.save(ckpt if wrapped else sd, dst_path)
    sys.exit(0)

for k in keys_to_remove:
    del sd[k]

# 保存
if wrapped:
    ckpt["state_dict"] = sd
    torch.save(ckpt, dst_path)
else:
    torch.save(sd, dst_path)

print("Saved pruned checkpoint to:", dst_path)
print("You can re-run your train command pointing to this new ckpt (or replace filename in config).")
