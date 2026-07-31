"""Load q_weights.pt from a run, produce static float weights, save to same run dir. Run from the repo root."""

from pathlib import Path

import argparse
import torch
from qt_pinn.qnn_generator import QuantumWeightGenerator
from qt_pinn.learned_proj.qnn_generator import QuantumWeightGeneratorLP


def _latest(base: Path = Path("checkpoints")) -> str:
    runs = sorted(base.glob("run_*"))
    if not runs:
        raise FileNotFoundError("No runs found in checkpoints/")
    return runs[-1].name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="latest",
                        help="run ID (e.g. run_0001) or 'latest'")
    parser.add_argument("--lp", action="store_true",
                        help="use learned-projection generator (train_lp.py runs)")
    args = parser.parse_args()

    run_id  = _latest() if args.run == "latest" else args.run
    run_dir = Path("checkpoints") / run_id

    gen = QuantumWeightGeneratorLP() if args.lp else QuantumWeightGenerator()
    gen.load_state_dict(torch.load(run_dir / "q_weights.pt", map_location="cpu", weights_only=True))
    gen.eval()

    with torch.no_grad():
        weights = gen()

    static = {k: v.detach().clone() for k, v in weights.items()}
    out    = run_dir / "static_weights.pt"
    torch.save(static, out)

    print(f"[{run_id}] Saved static_weights.pt")
    for k, v in static.items():
        print(f"  {k}: {v.shape}  {v.dtype}")


if __name__ == "__main__":
    main()
