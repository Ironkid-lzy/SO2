"""T001 engineering helper: inspect the official SO2 checkpoint structure.

Read-only. Does not train. Only prints metadata so the report can state the
checkpoint identity and the expected key layout of the EDAC `_load_state_dict_learn`
conversion path.
"""

# NOTE (T001 layout): this helper lives in _so2_work/scripts/. Its outputs go to
#   _so2_work/logs/<name>/   and the official checkpoint lives at
#   _so2_work/assets/checkpoints/ckpt/halfcheetah-medium-replay-v2.ckpt
import argparse
import hashlib
import os
import sys

import torch


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to the .ckpt / .pth file")
    ap.add_argument("--sha256", action="store_true", help="also compute sha256")
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        print("MISSING:", args.ckpt)
        return 2
    size = os.path.getsize(args.ckpt)
    print(f"path={args.ckpt}")
    print(f"size_bytes={size} ({size / 1e6:.2f} MB)")
    if args.sha256:
        print(f"sha256={sha256(args.ckpt)}")

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"top_level_type={type(obj).__name__}")
    if isinstance(obj, dict):
        print("top_level_keys:")
        for k in obj:
            v = obj[k]
            if isinstance(v, dict):
                ks = list(v.keys())
                print(f"  {k}: dict[{len(ks)}] first={ks[:6]}")
            else:
                print(f"  {k}: {type(v).__name__}")

    def walk(d, prefix="", depth=0):
        if depth > 2 or not isinstance(d, dict):
            return
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{'  ' * (depth + 1)}{prefix}{k}/ ({len(v)} entries)")
                walk(v, prefix + k + "/", depth + 1)
            elif isinstance(v, torch.Tensor):
                print(f"{'  ' * (depth + 1)}{prefix}{k}: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"{'  ' * (depth + 1)}{prefix}{k}: {type(v).__name__}")

    if isinstance(obj, dict):
        print("structure (depth<=3):")
        walk(obj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
