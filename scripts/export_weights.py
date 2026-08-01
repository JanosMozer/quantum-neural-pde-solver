"""Load q_weights.pt from a run, produce static float weights, save to same run dir. Run from the repo root."""

from pathlib import Path

import argparse
import json
import torch
from qt_pinn.qnn_generator import QuantumWeightGenerator
from qt_pinn.learned_proj.qnn_generator import QuantumWeightGeneratorLP


def _latest(base: Path = Path("checkpoints")) -> str:
    runs = sorted(base.glob("run_*"))
    if not runs:
        raise FileNotFoundError("No runs found in checkpoints/")
    return runs[-1].name


def _is_learned_proj(run_dir: Path) -> bool:
    """Auto-detect generator type from the run's own config.json (written by
    train_lp.py / train_gpu.py as "generator": "learned_proj[_gpu]")."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return False
    return "learned_proj" in json.loads(config_path.read_text()).get("generator", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="latest",
                        help="run ID (e.g. run_0001) or 'latest'")
    parser.add_argument("--lp", action="store_true",
                        help="force learned-projection generator (auto-detected from config.json otherwise)")
    args = parser.parse_args()

    run_id  = _latest() if args.run == "latest" else args.run
    run_dir = Path("checkpoints") / run_id

    use_lp = args.lp or _is_learned_proj(run_dir)
    gen = QuantumWeightGeneratorLP() if use_lp else QuantumWeightGenerator()
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
